from abc import ABC
import re, time
from enum import Enum, auto
from typing import Any, Optional, TypeAlias
import threading
import contextvars

from typego.utils import log_error

SKILL_ARG_TYPE: TypeAlias = int | float | str
SKILL_RET_TYPE: TypeAlias = Optional[int | float | bool | str]

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
    def __init__(self):
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
    def __init__(self, name: str, subsystem: SubSystem, thread_id: int, control: SkillControl):
        self.name = name
        self.subsystem = subsystem
        self.thread_id = thread_id
        self.control = control
        self.started_at = time.time()

class SkillRegistry:
    def __init__(self):
        self._items: dict[str, "SkillItem"] = {}
        self._funcs: dict[str, callable] = {}
        self._subsystems: dict[str, SubSystem] = {}

        # Reentrant locks per subsystem enable nested same-thread calls
        self._locks: dict[SubSystem, threading.RLock] = {
            ss: threading.RLock() for ss in SubSystem
        }
        self._running: dict[SubSystem, Optional[str]] = {ss: None for ss in SubSystem}
        self._active: dict[SubSystem, Optional[SkillExecution]] = {ss: None for ss in SubSystem}

        # Guard for mutating _active (lightweight; separate from subsystem locks)
        self._active_guard = threading.RLock()

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
    # Control APIs
    # -------------------------
    def pause(self, subsystem: Optional[SubSystem] = None) -> bool:
        with self._active_guard:
            targets = [subsystem] if subsystem else list(self._active.keys())
            ok = False
            for ss in targets:
                exe = self._active.get(ss)
                if exe:
                    exe.control.pause_event.set()
                    ok = True
            return ok

    def resume(self, subsystem: Optional[SubSystem] = None) -> bool:
        with self._active_guard:
            targets = [subsystem] if subsystem else list(self._active.keys())
            ok = False
            for ss in targets:
                exe = self._active.get(ss)
                if exe:
                    exe.control.pause_event.clear()
                    ok = True
            return ok

    def stop(self, subsystem: Optional[SubSystem] = None) -> bool:
        with self._active_guard:
            targets = [subsystem] if subsystem else list(self._active.keys())
            print(f"[SkillRegistry] Stopping subsystems: {[ss.name for ss in targets]}")
            ok = False
            for ss in targets:
                exe = self._active.get(ss)
                if exe:
                    exe.control.stop_event.set()
                    ok = True
            return ok

    # -------------------------
    # Execute with BUSY + control
    # -------------------------
    def execute(
        self,
        func_call: str,
        args: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Execute a registered skill by name with typed args.
        - Ensures exclusive subsystem execution (non-blocking BUSY).
        - Provides stop/pause control via threading.Event.
        - Allows nested same-thread calls (RLock)."""
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

        print(f"[SkillRegistry] Executing skill '{name}' with args {arg_list or kwargs}")

        item = self._items.get(name)
        if not item:
            return {"ok": False, "error": f"unknown skill '{name}'"}

        subsystem = self._subsystems.get(name, SubSystem.DEFAULT)
        lock = self._locks[subsystem]

        # Try to acquire the subsystem lock WITHOUT blocking
        if not lock.acquire(blocking=False):
            running = self._running.get(subsystem)
            print(f"[SkillRegistry] Subsystem {subsystem.name} is BUSY running '{running}'")
            return {"ok": False, "error": "BUSY", "subsystem": subsystem.name, "running": running}

        control = SkillControl()
        exe = SkillExecution(
            name=name,
            subsystem=subsystem,
            thread_id=threading.get_ident(),
            control=control
        )

        try:
            self._running[subsystem] = name
            with self._active_guard:
                self._active[subsystem] = exe

            # Prepare typed/ordered args
            parsed = item.parse_args(arg_list) if arg_list else item.parse_args([])

            # Pass control events iff the skill accepts them explicitly
            fn = self._funcs[name]
            if getattr(fn, "__accepts_stop__", False):
                kwargs["stop_event"] = control.stop_event
            if getattr(fn, "__accepts_pause__", False):
                kwargs["pause_event"] = control.pause_event

            # Set runtime context so helpers work even if function signature doesn't accept events
            token = _current_control.set(control)
            try:
                ret = fn(*parsed, **kwargs) if kwargs else fn(*parsed)
            finally:
                _current_control.reset(token)

            return {"ok": True, "data": ret}
        except Exception as e:
            return {"ok": False, "error": str(e)}
        finally:
            with self._active_guard:
                self._active[subsystem] = None
            self._running[subsystem] = None
            lock.release()


class SkillArg:
    def __init__(self, arg_name: str, arg_type: type):
        self.arg_name = arg_name
        self.arg_type = arg_type
    
    def __repr__(self) -> str:
        return f"{self.arg_name}: {self.arg_type.__name__}"

class SkillItem(ABC):
    _INTERNAL_ARGS = {"stop_event", "pause_event"}  # filtered out
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

    # def register_args(self, params: dict):
    #     for k, v in params.items():
    #         self._args.append(SkillArg(k, v))
    def register_args(self, params: dict | None):
        """Register argument definitions, skipping internal control args."""
        self._args.clear()
        if not params:
            return
        for n, t in params.items():
            if n not in self._INTERNAL_ARGS:
                self._args.append(SkillArg(n, t))

    def parse_args(self, args_str_list: list[SKILL_ARG_TYPE], allow_positional_args: bool = False) -> list[SKILL_ARG_TYPE]:
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
            # Allow positional arguments
            if arg.startswith('$') and allow_positional_args:
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
    

def evaluate_value(s: str) -> SKILL_RET_TYPE:
    if not s:  # Empty string or None
        return None
    
    # Strip whitespace once at the beginning
    s_clean = s.strip()
    
    # Check for None
    if s_clean == 'None':
        return None
    
    # Check for boolean values
    if s_clean == 'True':
        return True
    if s_clean == 'False':
        return False
    
    # Check for numeric values
    if s_clean.startswith(('-', '+')):
        num_str = s_clean[1:]
        sign = -1 if s_clean[0] == '-' else 1
    else:
        num_str = s_clean
        sign = 1
    
    # Check if it's a valid number
    if num_str.replace('.', '', 1).isdigit():
        if '.' in num_str:
            return sign * float(num_str)
        else:
            return sign * int(num_str)
    
    # Return original string if no conversion applies
    return s