import time
from dataclasses import dataclass
from typing import Optional
from typego.robot_info import RobotInfo
import json

STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"
STATUS_IN_PROGRESS = "in_progress"
STATUS_PENDING = "pending"

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

@dataclass
class SubtaskItem:
    start: float
    end: float
    content: str
    actions: list[ActionItem]
    status: str

    def finish_action(self, result: bool):
        if not self.actions:
            return
        
        action = self.actions[-1]
        action.end = time.time()
        action.status = STATUS_SUCCESS if result else STATUS_FAILED

    def execute_action(self, action: str):
        action_item = ActionItem(start=time.time(), end=0.0, content=action, status=STATUS_IN_PROGRESS)
        self.actions.append(action_item)

    def get_in_progress_action(self) -> ActionItem | None:
        if not self.actions:
            return None
        for action in self.actions:
            if action.status == STATUS_IN_PROGRESS:
                return action
        return None

    def _sim(self) -> dict:
        js = {
            "start": format_time(self.start),
            "end": format_time(self.end),
            "subtask": self.content,
            "status": self.status
        }
        return js
    
    def _full(self) -> dict:
        js = {
            "start": format_time(self.start),
            "end": format_time(self.end),
            "subtask": self.content,
            "status": self.status,
            "actions": [action._full() for action in self.actions]
        }
        return js
    
@dataclass
class InstructionItem:
    start: float
    end: float
    content: str
    plan: list[SubtaskItem]
    status: str

    def new(content: str):
        return InstructionItem(
            start=time.time(),
            end=0.0,
            content=content,
            plan=[],
            status=STATUS_PENDING
        )
    
    def idle():
        return InstructionItem(
            start=time.time(),
            end=0.0,
            content="Idle",
            plan=[],
            status=STATUS_IN_PROGRESS
        )
    
    def is_idle(self) -> bool:
        return self.content == "Idle"

    def set_plan(self, plan: list[str], interrupt: bool = False):
        new_plan = [SubtaskItem(start=0.0, end=0.0, content=subtask, actions=[], status=STATUS_PENDING) for subtask in plan]
        if interrupt:
            self.plan = new_plan
        else:
            self.plan.extend(new_plan)

    def get_latest_subtask(self) -> SubtaskItem | None:
        for subtask in self.plan:
            if subtask.status == STATUS_IN_PROGRESS:
                return subtask
            elif subtask.status == STATUS_PENDING:
                subtask.status = STATUS_IN_PROGRESS
                subtask.start = time.time()
                return subtask
        return None

    def finish_subtask(self, result: bool):
        if len(self.plan) == 0:
            print("No subtasks to finish.")
            return
        subtask = self.get_latest_subtask()
        if subtask:
            subtask.end = time.time()
            subtask.status = STATUS_SUCCESS if result else STATUS_FAILED

        if all(subtask.status != STATUS_IN_PROGRESS for subtask in self.plan):
            self.end = time.time()
            # TODO: maybe different status for the instruction itself
            self.status = STATUS_SUCCESS if result else STATUS_FAILED

    def is_finished(self) -> bool:
        return self.status in (STATUS_SUCCESS, STATUS_FAILED)

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
            "plan": [subtask._sim() for subtask in self.plan]
        }
        return js
    
class RobotMemory:
    def __init__(self, robot_info: RobotInfo):
        self.robot_info = robot_info
        self.history_inst: list[InstructionItem] = []
        self.current_inst: InstructionItem = InstructionItem.idle()
        self.default_subtask: SubtaskItem = SubtaskItem(
            start=0.0,
            end=0.0,
            content="None",
            actions=[],
            status=STATUS_IN_PROGRESS
        )

        self.in_progress_action: Optional[ActionItem] = None

    def get_history_inst_str(self) -> str:
        rslt = "["
        in_progress = False
        for item in self.history_inst:
            if item.status == STATUS_IN_PROGRESS:
                in_progress = True
            rslt += str(item._sim())
        if not in_progress:
            rslt += str(self.current_inst._sim())
        rslt += "]"
        return rslt
    
    def get_current_plan_str(self) -> str:
        rslt = "["
        for subtask in self.current_inst.plan:
            rslt += str(subtask._sim())
        rslt += "]"
        return rslt

    def get_history_action_str(self) -> str:
        subt = self.get_subtask()
        rslt = "["
        for a in subt.actions:
            rslt += str(a._full())

        if len(subt.actions) == 0 and self.in_progress_action:
            rslt += str(self.in_progress_action._full())
        rslt += "]"
        return rslt

    def new_instruction(self, inst: str | None):
        if inst:
            self.history_inst.append(InstructionItem.new(inst))

        if self.current_inst.is_idle():
            for item in self.history_inst:
                if not item.is_finished():
                    self.current_inst = item
                    self.current_inst.status = STATUS_IN_PROGRESS
                    break

    def process_s2_response(self, response: str):
        if response.startswith('```json'):
            response = response[7:-3].strip()

        js = json.loads(response)
        interrupt = js.get('interrupt_current_task', False)
        print(f"[Memory] Processing S2 response: {self.current_inst.content}, {self.current_inst.plan}")
        self.current_inst.set_plan(js['new_plan'], interrupt)

    def get_subtask(self) -> SubtaskItem:
        subt = self.current_inst.get_latest_subtask()
        return subt if subt else self.default_subtask

    def end_subtask(self, result: bool):
        self.current_inst.finish_subtask(result)
        if self.current_inst.is_finished():
            self.current_inst = InstructionItem.idle()

    def execute_action(self, action: str):
        self.get_subtask().execute_action(action)
        self.in_progress_action = self.get_subtask().get_in_progress_action()

    def finish_action(self, result: bool):
        self.get_subtask().finish_action(result)