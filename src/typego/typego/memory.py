import time
from dataclasses import dataclass
from typing import Optional
from typego.robot_info import RobotInfo
import json

STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"
STATUS_IN_PROGRESS = "in_progress"
STATUS_PENDING = "pending"
STATUS_STOPPED = "stopped"

def format_time(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    if seconds <= 0:
        return "N/A"
    return time.strftime("%H:%M:%S", time.localtime(seconds))

@dataclass
class ActionItem:
    start: float
    end: float
    content: str
    status: str

    def _full(self) -> dict:
        js = {
            "start": format_time(self.start),
            "end": format_time(self.end),
            "action": self.content,
            "status": self.status
        }
        return js
    
    def is_finished(self):
        """Mark the action as finished with a result."""
        return self.status in (STATUS_SUCCESS, STATUS_FAILED, STATUS_STOPPED)
    
    def finish(self, result: bool):
        """Finish the action with a result."""
        if self.status == STATUS_IN_PROGRESS:
            self.end = time.time()
            self.status = STATUS_SUCCESS if result else STATUS_FAILED
    
@dataclass
class InstructionItem:
    start: float
    end: float
    content: str
    plan: str | None
    actions: list[ActionItem]
    status: str

    def new(content: str):
        return InstructionItem(
            start=0.0,
            end=0.0,
            content=content,
            plan=None,
            actions=[],
            status=STATUS_PENDING
        )
    
    def idle():
        return InstructionItem(
            start=time.time(),
            end=0.0,
            content="Idle",
            plan=None,
            actions=[],
            status=STATUS_IN_PROGRESS
        )
    
    def is_idle(self) -> bool:
        return self.content == "Idle"
    
    def is_finished(self) -> bool:
        return self.status in (STATUS_SUCCESS, STATUS_FAILED)

    def set_plan(self, plan: str, interrupt: bool = False):
        plan = plan.strip()
        if not plan:
            plan = None
        self.plan = plan
        self.actions = []

    def get_plan(self) -> str:
        if self.plan:
            return '\n```\n' + self.plan + '\n```\n'
        return "None"

    def add_action(self, action: str):
        action_item = ActionItem(start=0, end=0.0, content=action, status=STATUS_PENDING)
        self.actions.append(action_item)

    def get_in_progress_action(self) -> ActionItem | None:
        if not self.actions:
            return None
        for action in self.actions:
            if action.status == STATUS_IN_PROGRESS:
                return action
            elif action.status == STATUS_PENDING:
                action.status = STATUS_IN_PROGRESS
                action.start = time.time()
                return action
        return None

    # def finish_action(self, result: bool):
    #     if not self.actions or not any(action.status == STATUS_IN_PROGRESS for action in self.actions):
    #         raise ValueError("Cannot finish action: no actions in progress.")

    #     action = self.get_in_progress_action()
    #     if action:
    #         action.end = time.time()
    #         action.status = STATUS_SUCCESS if result else STATUS_FAILED

    def has_unfinished_action(self) -> bool:
        if not self.actions:
            return False
        if any(not action.is_finished() for action in self.actions):
            return True
        return False

    def _sim(self) -> dict:
        js = {
            "start": format_time(self.start),
            "end": format_time(self.end),
            "inst": self.content,
            "status": self.status
        }
        return js

    def _full(self) -> dict:
        js = {
            "start": format_time(self.start),
            "end": format_time(self.end),
            "inst": self.content,
            "status": self.status,
            "plan": self.plan if self.plan else "None",
        }
        return js
    
class RobotMemory:
    def __init__(self, robot_info: RobotInfo):
        self.robot_info = robot_info
        self.inst_list: list[InstructionItem] = []
        
        self.in_progress_action: Optional[ActionItem] = None

        self.idle_inst = InstructionItem.idle()
        self.current_inst: InstructionItem = self.idle_inst

    def get_inst_list_str(self) -> str:
        rslt = "["
        in_progress = False
        for item in self.inst_list:
            if item.status == STATUS_IN_PROGRESS:
                in_progress = True
            rslt += str(item._sim()) + ',\n'

        # no active instruction, append the Idle instruction
        if not in_progress:
            rslt += str(self.current_inst._sim())
        rslt += "]"
        return rslt
    
    def get_current_inst_str(self) -> str:
        if self.current_inst.is_idle():
            return "Idle"
        return self.current_inst.content
    
    def get_current_plan_str(self) -> str:
        return self.current_inst.get_plan()

    def get_history_action_str(self) -> str:
        rslt = "["
        for a in self.current_inst.actions:
            rslt += str(a._full()) + ',\n'

        if len(self.current_inst.actions) == 0 and self.in_progress_action:
            rslt += str(self.in_progress_action._full())
        rslt += "]"
        return rslt

    def process_s2_response(self, response: str):
        if response.startswith('```json'):
            response = response[7:-3].strip()

        js = json.loads(response)
        interrupt = js.get('interrupt_current_task', False)
        print(f"[Memory] Processing S2 response: {self.current_inst.content}, {self.current_inst.plan}")
        self.current_inst.set_plan(js['new_plan'], interrupt)

    def add_inst(self, inst: str | None):
        if not inst:
            return

        inst = inst.strip()
        if inst:
            self.inst_list.append(InstructionItem.new(inst))

        if self.current_inst.is_idle():
            # pick the first unfinished instruction
            for item in self.inst_list:
                if not item.is_finished():
                    self.current_inst = item
                    self.current_inst.start = time.time()
                    self.current_inst.status = STATUS_IN_PROGRESS
                    return

    def end_inst(self, result: bool):
        if self.current_inst.has_unfinished_action():
            raise ValueError("Cannot end instruction with unfinished actions.")
        
        self.current_inst.end = time.time()
        self.current_inst.status = STATUS_SUCCESS if result else STATUS_FAILED

        self.current_inst = self.idle_inst

    def add_action(self, action: str) -> ActionItem | None:
        self.current_inst.add_action(action)
        self.in_progress_action = self.current_inst.get_in_progress_action()
        return self.in_progress_action

    def stop_action(self):
        if self.in_progress_action and self.in_progress_action.status == STATUS_IN_PROGRESS:
            self.in_progress_action.status = STATUS_STOPPED
            self.in_progress_action.end = time.time()
            self.in_progress_action = self.current_inst.get_in_progress_action()
        else:
            print("No action in progress to stop.")