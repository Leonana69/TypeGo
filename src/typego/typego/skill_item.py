from abc import ABC
import re, time
from enum import Enum, auto
from typing import Any, Optional
import threading
import contextvars
import uuid
import itertools

from typego.utils import log_error, print_t


# ----------------------------
# Robot Subsystems
# ----------------------------
class SubSystem(Enum):
    DEFAULT = auto()
    MOVEMENT = auto()
    SOUND = auto()

# =========================
# Runtime control context
# =========================
class SkillControl:
    def __init__(self, task_id: int):
        self.task_id = task_id
        self.stop_event = threading.Event()
        # Pause policy: pause_event SET => paused, CLEAR => running
        self.pause_event = threading.Event()

_current_control: contextvars.ContextVar[Optional[SkillControl]] = contextvars.ContextVar(
    "skill_control", default=None
)

def current_stop_event() -> Optional[threading.Event]:
    ctl = _current_control.get()
    return ctl.stop_event if ctl else None

def current_pause_event() -> Optional[threading.Event]:
    ctl = _current_control.get()
    return ctl.pause_event if ctl else None

def wait_if_paused(sleep_s: float = 0.02):
    """Cooperative pause gate. Call this inside long loops."""
    ctl = _current_control.get()
    if not ctl:
        return
    # Sleep in short slices so stop can preempt a long pause quickly
    while ctl.pause_event.is_set() and not ctl.stop_event.is_set():
        time.sleep(sleep_s)

# =========================
# Skill registry & execution
# =========================
class SkillExecution:
    """Bookkeeping for an active skill."""
    def __init__(self, exec_id: str, name: str, subsystem: SubSystem, thread: threading.Thread, control: SkillControl, task_id: int):
        self.id = exec_id
        self.name = name
        self.subsystem = subsystem
        self.thread = thread
        self.control = control
        self.task_id = task_id
        self.started_at = time.time()

    def is_alive(self) -> bool:
        return self.thread.is_alive()

    def has_finished(self) -> bool:
        return not self.thread.is_alive()

class SkillRegistry:
    def __init__(self):
        self._items: dict[str, "SkillItem"] = {}
        self._funcs: dict[str, callable] = {}
        self._subsystems: dict[str, SubSystem] = {}

        self._locks: dict[SubSystem, threading.Semaphore] = {
            ss: threading.Semaphore(1) for ss in SubSystem
        }
        self._running: dict[SubSystem, Optional[str]] = {ss: None for ss in SubSystem}
        self._active: dict[SubSystem, Optional[SkillExecution]] = {ss: None for ss in SubSystem}
        self._paused: dict[SubSystem, Optional[SkillExecution]] = {ss: None for ss in SubSystem}

        self._active_guard = threading.RLock()

        # ID generator
        self._exec_counter = itertools.count(1)
        self._executions: dict[str, SkillExecution] = {}  # id → execution
        self._task_to_exec: dict[int, str] = {}  # task_id → exec_id
        self._task_controls: dict[int, SkillControl] = {}  # task_id → control

    def register(self, name: str, description: str = "",
                 params: dict | None = None,
                 subsystem: SubSystem = SubSystem.DEFAULT):
        def deco(fn):
            item = SkillItem(name=name, description=description)
            item.register_args(params)
            self._items[name] = item
            self._funcs[name] = fn
            self._subsystems[name] = subsystem
            return fn
        return deco

    def names(self) -> list[str]:
        return list(self._items.keys())

    def get_skill_list(self, keys: list[str]=[]) -> list[str]:
        """Returns a list of target registered skills in string format."""
        if keys:
            l = []
            for key in keys:
                if key in self._items:
                    l.append(str(self._items[key]))
                else:
                    log_error(f"Skill '{key}' not found in registry.")
            return l
        return [str(item) for item in self._items.values()]

    # -------------------------
    # Control APIs - Now Task-Specific
    # -------------------------
    def pause(self, task_id: Optional[int] = None, subsystem: Optional[SubSystem] = None) -> bool:
        """
        Pause a specific task by task_id, or all tasks in a subsystem.
        If neither specified, pauses all active tasks.
        """
        print_t(f"[SkillRegistry] Pausing task_id={task_id}, subsystem={subsystem}")
        with self._active_guard:
            if task_id is not None:
                # Pause specific task
                control = self._task_controls.get(task_id)
                if not control:
                    print_t(f"[SkillRegistry] Task {task_id} not found or already finished")
                    return False
                
                exec_id = self._task_to_exec.get(task_id)
                exe = self._executions.get(exec_id) if exec_id else None
                if not exe:
                    return False
                
                fn = self._funcs.get(exe.name)
                if getattr(fn, "__accepts_pause__", False):
                    control.pause_event.set()
                    print_t(f"[SkillRegistry] Paused task {task_id} (skill '{exe.name}')")
                    
                    # Mark subsystem free and move to paused
                    ss = exe.subsystem
                    self._paused[ss] = exe
                    self._active[ss] = None
                    self._running[ss] = None
                    self._locks[ss].release()
                    return True
                else:
                    print_t(f"[SkillRegistry] Task {task_id} (skill '{exe.name}') does not support pause, stopping instead")
                    control.stop_event.set()
                    return True
            else:
                # Pause all tasks in subsystem(s)
                targets = [subsystem] if subsystem else list(self._active.keys())
                ok = False
                for ss in targets:
                    exe = self._active.get(ss)
                    if not exe:
                        continue
                    
                    control = self._task_controls.get(exe.task_id)
                    if not control:
                        continue
                    
                    fn = self._funcs.get(exe.name)
                    if getattr(fn, "__accepts_pause__", False):
                        control.pause_event.set()
                        print_t(f"[SkillRegistry] Paused task {exe.task_id} (skill '{exe.name}') in subsystem {ss.name}")
                        
                        self._paused[ss] = exe
                        self._active[ss] = None
                        self._running[ss] = None
                        self._locks[ss].release()
                    else:
                        control.stop_event.set()
                        print_t(f"[SkillRegistry] Task {exe.task_id} does not support pause, stopping instead")
                    ok = True
                return ok

    def resume(self, task_id: Optional[int] = None, subsystem: Optional[SubSystem] = None) -> bool:
        """
        Resume a specific task by task_id, or all paused tasks in a subsystem.
        If neither specified, resumes all paused tasks.
        """
        print_t(f"[SkillRegistry] Resuming task_id={task_id}, subsystem={subsystem}")
        with self._active_guard:
            if task_id is not None:
                # Resume specific task
                control = self._task_controls.get(task_id)
                if not control:
                    print_t(f"[SkillRegistry] Task {task_id} not found")
                    return False
                
                exec_id = self._task_to_exec.get(task_id)
                exe = self._executions.get(exec_id) if exec_id else None
                if not exe:
                    return False
                
                ss = exe.subsystem
                if self._paused.get(ss) != exe:
                    print_t(f"[SkillRegistry] Task {task_id} is not currently paused")
                    return False
                
                control.pause_event.clear()
                self._active[ss] = exe
                self._running[ss] = exe.name
                self._paused[ss] = None
                self._locks[ss].acquire()
                print_t(f"[SkillRegistry] Resumed task {task_id} (skill '{exe.name}')")
                return True
            else:
                # Resume all paused tasks in subsystem(s)
                targets = [subsystem] if subsystem else list(self._paused.keys())
                ok = False
                for ss in targets:
                    exe = self._paused.get(ss)
                    if not exe:
                        continue
                    
                    control = self._task_controls.get(exe.task_id)
                    if not control:
                        continue
                    
                    control.pause_event.clear()
                    self._active[ss] = exe
                    self._running[ss] = exe.name
                    self._paused[ss] = None
                    self._locks[ss].acquire()
                    print_t(f"[SkillRegistry] Resumed task {exe.task_id} (skill '{exe.name}') in subsystem {ss.name}")
                    ok = True
                return ok

    def stop(self, task_id: Optional[int] = None, subsystem: Optional[SubSystem] = None, timeout: float | None = None) -> bool:
        """
        Stop a specific task by task_id, or all tasks in a subsystem.
        If neither specified, stops all active tasks.
        """
        print_t(f"[SkillRegistry] Stopping task_id={task_id}, subsystem={subsystem}")
        if task_id is not None:
            # Stop specific task
            with self._active_guard:
                control = self._task_controls.get(task_id)
                if not control:
                    print_t(f"[SkillRegistry] Task {task_id} not found or already finished")
                    return False
                
                exec_id = self._task_to_exec.get(task_id)
                exe = self._executions.get(exec_id) if exec_id else None
            
            if exe:
                print_t(f"[SkillRegistry] Stopping task {task_id} (skill '{exe.name}')")
                control.stop_event.set()
                exe.thread.join(timeout=timeout)
                return True
            return False
        else:
            # Stop all tasks in subsystem(s)
            targets = [subsystem] if subsystem else list(self._active.keys())
            ok = False
            for ss in targets:
                with self._active_guard:
                    exe = self._active.get(ss)
                if not exe:
                    continue
                
                control = self._task_controls.get(exe.task_id)
                if control:
                    print_t(f"[SkillRegistry] Stopping task {exe.task_id} in subsystem {ss.name}")
                    control.stop_event.set()
                    exe.thread.join(timeout=timeout)
                    ok = True
            return ok

    def execute(
        self,
        func_call: str,
        args: dict[str, Any] | None = None,
        task_id: int = -1,
        callback: Optional[callable] = None
    ) -> dict[str, Any]:
        """Execute a registered skill asynchronously in a background thread.
        - Ensures exclusive subsystem execution (non-blocking BUSY).
        - Provides stop/pause control via threading.Event per task.
        - Allows nested same-thread calls.
        - Overrides any existing execution with the same task_id.
        """

        # ---- Parse call ----
        if not args:
            m = re.match(r"(\w+)\((.*)\)", func_call)
            if not m:
                return {"ok": False, "error": f"invalid function call '{func_call}'"}
            name = m.group(1)
            arg_list = [a.strip() for a in m.group(2).split(",") if a.strip()]
            kwargs = {}
        else:
            name = func_call
            arg_list = list(args.values())
            kwargs = dict(args)

        print_t(f"[SkillRegistry] Executing skill '{name}' with args {arg_list or kwargs} for task_id={task_id}")

        if name == "continue":
            return {"ok": True, "id": "continue"}

        item = self._items.get(name)
        if not item:
            return {"ok": False, "error": f"unknown skill '{name}'"}

        subsystem = self._subsystems.get(name, SubSystem.DEFAULT)

        # ---- Check for existing task with same task_id ----
        old_exe = None
        old_control = None
        with self._active_guard:
            if task_id in self._task_to_exec:
                old_exec_id = self._task_to_exec[task_id]
                old_exe = self._executions.get(old_exec_id)
                old_control = self._task_controls.get(task_id)
                if old_exe and old_exe.is_alive() and old_control:
                    print_t(f"[SkillRegistry] Task {task_id} already running as '{old_exe.name}' [{old_exec_id}], stopping it")
                    old_control.stop_event.set()
        
        # Wait for old task outside the lock to avoid deadlock
        if old_exe and old_exe.is_alive():
            old_exe.thread.join(timeout=2.0)
            if old_exe.is_alive():
                print_t(f"[SkillRegistry] Warning: Old task {task_id} did not stop in time")

        # ---- Check if subsystem is available ----
        lock = self._locks[subsystem]
        if not lock.acquire(blocking=False):
            with self._active_guard:
                current_exe = self._active.get(subsystem)
                current_task = current_exe.task_id if current_exe else None
            print_t(f"[SkillRegistry] Subsystem {subsystem.name} is BUSY (task_id={current_task})")
            return {"ok": False, "error": f"subsystem {subsystem.name} is busy", "busy": True, "current_task_id": current_task}

        # Create task-specific control
        control = SkillControl(task_id)

        # Generate unique ID
        exec_id = f"{name}-{next(self._exec_counter)}-{uuid.uuid4().hex[:6]}"

        def runner(callback: Optional[callable] = None):
            token = _current_control.set(control)
            try:
                fn = self._funcs[name]

                if getattr(fn, "__accepts_stop__", False):
                    kwargs["stop_event"] = control.stop_event
                if getattr(fn, "__accepts_pause__", False):
                    kwargs["pause_event"] = control.pause_event
                if getattr(fn, "__accepts_task_id__", False):
                    kwargs["task_id"] = task_id

                parsed = item.parse_args(arg_list) if arg_list else []
                print_t(f"[SkillRegistry] Started skill '{name}' [{exec_id}] for task_id={task_id} in subsystem {subsystem.name}")
                ret = fn(*parsed, **kwargs) if kwargs else fn(*parsed)
                if callback:
                    callback(ret)
            except Exception as e:
                print_t(f"[SkillRegistry] ERROR in skill '{name}': {e}")
            finally:
                with self._active_guard:
                    self._active[subsystem] = None
                    # Clean up task mapping and control
                    if self._task_to_exec.get(task_id) == exec_id:
                        del self._task_to_exec[task_id]
                    if task_id in self._task_controls:
                        del self._task_controls[task_id]
                self._running[subsystem] = None
                lock.release()
                _current_control.reset(token)
                print_t(f"[SkillRegistry] Skill '{name}' [{exec_id}] for task_id={task_id} finished, subsystem {subsystem.name} released")

        t = threading.Thread(target=runner, daemon=True, args=(callback,))
        exe = SkillExecution(exec_id, name, subsystem, t, control, task_id)

        with self._active_guard:
            self._active[subsystem] = exe
            self._executions[exec_id] = exe
            self._task_to_exec[task_id] = exec_id
            self._task_controls[task_id] = control  # Store task-specific control
        self._running[subsystem] = name

        t.start()
        return {"ok": True, "id": exec_id, "task_id": task_id}
    
    def get_status(self, exec_id: str = None, task_id: int = None) -> dict[str, Any]:
        """Get status by execution ID or task ID."""
        if task_id is not None:
            exec_id = self._task_to_exec.get(task_id)
            if not exec_id:
                return {"ok": False, "error": f"unknown task id {task_id}"}
        
        exe = self._executions.get(exec_id)
        if not exe:
            return {"ok": False, "error": f"unknown execution id {exec_id}"}
        
        control = self._task_controls.get(exe.task_id)
        return {
            "ok": True,
            "id": exe.id,
            "name": exe.name,
            "subsystem": exe.subsystem.name,
            "task_id": exe.task_id,
            "alive": exe.is_alive(),
            "paused": control.pause_event.is_set() if control else False,
            "stopped": control.stop_event.is_set() if control else False,
            "started_at": exe.started_at,
        }

class SkillArg:
    def __init__(self, arg_name: str, arg_type: type):
        self.arg_name = arg_name
        self.arg_type = arg_type
    
    def __repr__(self) -> str:
        return f"{self.arg_name}: {self.arg_type.__name__}"

class SkillItem:
    _INTERNAL_ARGS = {"stop_event", "pause_event", "task_id"}  # filtered out
    def __init__(self, name: str, description: str):
        self._name: str = name
        self._description: str = description
        self._args: list[SkillArg] = []

    @property
    def name(self): return self._name

    @property
    def description(self): return self._description

    @property
    def args(self):
        # filter here so the internal args never leak out
        return tuple(arg for arg in self._args if arg.arg_name not in self._INTERNAL_ARGS)
    
    def __repr__(self) -> str:
        return (f"{self._name}: "
                f"args: {[arg for arg in self._args]}, "
                f"desc: {self._description}")

    def register_args(self, params: dict | None):
        """Register argument definitions, skipping internal control args."""
        self._args.clear()
        if not params:
            return
        for n, t in params.items():
            if n not in self._INTERNAL_ARGS:
                self._args.append(SkillArg(n, t))

    def parse_args(self, args_str_list: list[str]) -> list[Optional[int | float | bool | str]]:
        """Parses the string of arguments and converts them to the expected types."""
        # Check the number of arguments
        if len(args_str_list) != len(self.args):
            log_error(f"Func {self.name} expected {len(self.args)} arguments, but got {len(args_str_list)}.", raise_error=True)

        parsed_args = []
        for i, arg in enumerate(args_str_list):
            # if arg is not a string, skip parsing
            if not isinstance(arg, str):
                parsed_args.append(arg)
                continue
            
            try:
                if self.args[i].arg_type == bool:
                    parsed_args.append(arg.strip().lower() == 'true')
                else:
                    parsed_args.append(self.args[i].arg_type(arg.strip()))
            except ValueError as e:
                log_error(f"Error parsing argument {i + 1}. Expected type {self._args[i].arg_type.__name__}, but got value '{arg.strip()}'. Original error: {e}", raise_error=True)
        return parsed_args