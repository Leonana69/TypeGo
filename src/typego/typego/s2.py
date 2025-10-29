from dataclasses import dataclass
from datetime import datetime
import json, time
import re
from typing import Optional
from json import JSONEncoder

from typego.utils import print_t

STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"
STATUS_IN_PROGRESS = "in_progress"
STATUS_PAUSED = "paused"
STATUS_STOPPED = "stopped"

DEFAULT_PLAN_CONTENT = "Default bootstrap plan"

def format_time(ts):
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts else None

@dataclass
class ActionItem:
    start: float
    end: float
    state: str
    content: str
    status: str

    def __repr__(self) -> str:
        return json.dumps(self.to_dict(), indent=4)

    def to_dict(self):
        return {
            "start": format_time(self.start),
            "end": format_time(self.end),
            "action": self.content,
            "status": self.status
        }
    
    def is_finished(self):
        """Mark the action as finished with a result."""
        return self.status in (STATUS_SUCCESS, STATUS_FAILED, STATUS_STOPPED)
    
    def finish(self, result: bool):
        """Finish the action with a result."""
        if self.status == STATUS_IN_PROGRESS:
            self.end = time.time()
            self.status = STATUS_SUCCESS if result else STATUS_FAILED

@dataclass
class S2DPlanState:
    name: str
    action: str
    update: dict
    trans: list[str]

    def __repr__(self):
        return f"S2DPlanState(name={self.name}, action={self.action}, update={self.update}, trans={self.trans})"
    
    def to_dict(self):
        return self._raw_content

    def __init__(self, name: str, content: str | dict):
        self.name = name
        self._raw_content = content

        if isinstance(content, str):
            # Split once into "action" and "rest"
            try:
                action_part, rest = map(str.strip, content.split("->", 1))
            except ValueError:
                raise ValueError(f"Invalid content format: {content!r}")

            self.action = action_part
            self.update = {}

            # Handle transitions with optional "if"
            if "if" in rest:
                target, condition = map(str.strip, rest.split("if", 1))
                self.trans = [f"{target}: {condition}"]
            else:
                self.trans = [f"{rest}: always"]

        elif isinstance(content, dict):
            self.action = content.get("action", "")
            self.update = content.get("update", {})
            self.trans = list(content.get("trans", []))  # copy for safety

        else:
            raise TypeError(f"Unsupported content type: {type(content)}")

class S2PlanEncoder(JSONEncoder):
    """Custom JSON encoder for S2DPlanState class"""
    def default(self, obj):
        if isinstance(obj, S2DPlanState):
            return obj.to_dict()
        elif isinstance(obj, ActionItem):
            return obj.to_dict()
        return super().default(obj)

class S2DPlan:
    ID_COUNTER = 0
    PLAN_LIST: dict[int, "S2DPlan"] = {}
    def __init__(self, content: str):
        self.id = S2DPlan.ID_COUNTER
        S2DPlan.ID_COUNTER += 1

        self.variables = {}
        self.global_trans = []
        self.states: dict[str, S2DPlanState] = {
            "DEFAULT": S2DPlanState("DEFAULT", "Do anything that can help with the given task -> DEFAULT")
        }
        self.current_state = "DEFAULT"

        self.start_time = time.time()
        self.end_time = None
        self.content = content
        self.status = STATUS_IN_PROGRESS

        self.s2s_history: list[ActionItem] = []  # track actions in this plan

        S2DPlan.PLAN_LIST[self.id] = self

        # pause the DEFAULT plan if this is not it
        if self.id != 0:
            default_plan = S2DPlan.PLAN_LIST.get(0)
            if default_plan and default_plan.status == STATUS_IN_PROGRESS:
                default_plan.status = STATUS_PAUSED
    
    @classmethod
    def init_default(cls):
        """Initialize a default plan to track initial robot actions before any plan arrives."""
        default_plan = cls.__new__(cls)  # bypass __init__
        default_plan.id = 0
        cls.ID_COUNTER = 1

        default_plan.variables = {}
        default_plan.global_trans = [
            "CHASE_BALL: see a sports ball"
        ]
        default_plan.states = {
            "IDLE": S2DPlanState("IDLE", "standby, wait for instructions -> IDLE"),
            "CHASE_BALL": S2DPlanState("CHASE_BALL", "follow the sports ball -> IDLE")
        }
        default_plan.current_state = "IDLE"

        default_plan.start_time = time.time()
        default_plan.end_time = None
        default_plan.content = DEFAULT_PLAN_CONTENT
        default_plan.status = STATUS_IN_PROGRESS
        cls.PLAN_LIST[default_plan.id] = default_plan

    @classmethod
    def init_test_plan(cls):
        """Initialize a test plan for unit testing."""
        test_plan = cls("Test plan that makes the robot continuously do the same action.")
        test_plan.variables = {}
        test_plan.global_trans = []
        test_plan.current_state = "DO_ACTION"
        test_plan.states = {
            "DO_ACTION": S2DPlanState("DO_ACTION", "nod head once -> DO_ACTION"),
        }
        cls.PLAN_LIST[test_plan.id] = test_plan
        return test_plan

    def parse(self, plan_json: dict):
        self.variables = plan_json.get("variables", {})
        self.global_trans = plan_json.get("global_trans", [])
        self.current_state = plan_json.get("initial_state", "DEFAULT")
        self.states = {name: S2DPlanState(name, content) for name, content in plan_json.get("states", {}).items()}

        # Move action history in DEFAULT state to the new current state
        if self.s2s_history and self.s2s_history[0].state == "DEFAULT":
            for action in self.s2s_history:
                action.state = self.current_state

    def is_active(self):
        return self.status == STATUS_IN_PROGRESS or self.status == STATUS_PAUSED

    @classmethod
    def process_s2d_response(cls, llm_response: str) -> Optional["S2DPlan"]:
        def parse_task(task: str) -> list[tuple[str, Optional[int | str]]]:
            """
            Parse a task string like:
                'stop(5);update(2);new(task_content)'
            Returns a list of (command: str, argument: int | str | None) tuples.
            Raises ValueError if any command has invalid format.
            """
            results = []
            for segment in task.split(";"):
                segment = segment.strip()
                if not segment:
                    continue

                # Match command(arg)
                m = re.fullmatch(r"([a-zA-Z_]+)\((.*?)\)", segment)
                if not m:
                    raise ValueError(f"Malformed action: {segment} (expected format 'command(arg)')")

                command, inner = m.groups()
                inner = inner.strip()

                # Try to parse as int if numeric
                if inner == "":
                    arg: Optional[int | str] = None
                elif inner.isdigit():
                    arg = int(inner)
                else:
                    arg = inner.strip("'")  # treat as string

                results.append((command, arg))

            return results
        try:
            if "```json" in llm_response:
                json_content = llm_response.split("```json", 1)[1].rsplit("```", 1)[0]
            else:
                json_content = llm_response
            plan_json = json.loads(json_content)
        except Exception as e:
            raise ValueError(f"Failed to parse plan: {e}")

        commands = parse_task(plan_json.get("action", "").strip())  # validate format

        for command, task_arg in commands:
            if command == "new":
                if not isinstance(task_arg, str):
                    raise ValueError(f"Invalid task content for 'new': {task_arg}")
                plan = cls(task_arg)
                plan.parse(plan_json)
                return plan
            elif command == "update":
                plan = cls.PLAN_LIST.get(task_arg)
                if plan:
                    plan.parse(plan_json)
                else:
                    raise ValueError(f"Plan with ID {task_arg} not found for update.")
            elif command == "stop":
                plan = cls.PLAN_LIST.get(task_arg)
                if plan and plan.status == STATUS_IN_PROGRESS:
                    plan.status = STATUS_STOPPED
                    plan.end_time = time.time()
            elif command == "continue":
                plan = cls.PLAN_LIST.get(task_arg)
                if plan and plan.status != STATUS_IN_PROGRESS:
                    plan.status = STATUS_IN_PROGRESS
                    plan.end_time = None

                    # TODO: spawn s2s thread if not running
            else:
                raise ValueError(f"Unknown command: {command}")
        return None

    @classmethod
    def get_history_sorted_by_time(cls, after_time: float = None):
        plans = cls.PLAN_LIST.values()
        if after_time is not None:
            plans = filter(lambda p: p.start_time > after_time, plans)
        return sorted(plans, key=lambda p: p.start_time)

    def get_current_state(self):
        return self.current_state

    def transit(self, new_state: str):
        if new_state not in self.states and new_state != "DONE":
            raise ValueError(f"Invalid state: {new_state}")
        
        if self.current_state == new_state:
            self.update_local_data()
            return
        
        self.current_state = new_state

        if new_state == "DONE":
            self.status = STATUS_SUCCESS
            self.end_time = time.time()

    def process_s2s_response(self, response: str) -> str | None:
        if response.startswith('```json'):
            response = response[7:-3].strip()

        json_response = json.loads(response)
        next_action = json_response.get("action", None)
        if next_action and next_action != "continue()":
            self.add_action(next_action)

        next_state = json_response.get("trans", None)
        if next_state:
            self.transit(next_state)

        return next_action

    def add_action(self, action: str):
        """Add an action to the current state."""
        if self.status != STATUS_IN_PROGRESS:
            # Cannot add action to a non-active plan
            return

        action_item = ActionItem(start=time.time(), end=None, state=self.current_state, content=action, status=STATUS_IN_PROGRESS)
        self.s2s_history.append(action_item)

    def finish_action(self, result: bool):
        """Finish the last action with a result."""
        # TODO: fix this function

        print(f"[S2DPlan] Finishing action with result: {result}")

        if not self.s2s_history or not any(action.status == STATUS_IN_PROGRESS for action in self.s2s_history):
            raise ValueError("Cannot finish action: no actions in progress.")

        for action in reversed(self.s2s_history):
            if action.status == STATUS_IN_PROGRESS:
                print(f"[S2DPlan] Finishing action: {action}")
                action.finish(result)
                break

    def update_local_data(self):
        updates = self.states[self.current_state].update
        for key, expr in updates.items():
            try:
                self.variables[key] = eval(expr, {}, self.variables.copy())
            except Exception as e:
                print(f"Failed to evaluate update for key '{key}': {expr}")
                raise e
            
    def __repr__(self) -> str:
        if self.current_state == "DEFAULT":
            plan = "DEFAULT"
        else:
            plan = {
                "variables": self.variables,
                "global_trans": self.global_trans,
                "current_state": self.current_state,
                "states": self.states
            }

        ret = {
            "id": str(self.id),
            "start": format_time(self.start_time),
            "end": format_time(self.end_time),
            "content": self.content,
            "status": self.status,
        }

        if self.status == STATUS_IN_PROGRESS:
            ret["plan"] = plan

        return json.dumps(ret, indent=4, cls=S2PlanEncoder)

    @classmethod
    def get_s2d_input(cls) -> str:
        rslt = "Task list: \n"
        rslt += str(cls.get_history_sorted_by_time())
        return rslt

    def get_s2s_input(self) -> str:
        matching_actions = []
        for action in reversed(self.s2s_history):
            if action.state == self.current_state:
                matching_actions.insert(0, action)
            else:
                break
        info = {
            "variables": self.variables,
            "global_trans": self.global_trans,
            "current_state": {
                self.current_state: self.states[self.current_state],
            },

            "action_history": matching_actions
        }
        return json.dumps(info, indent=4, cls=S2PlanEncoder)

test_response = """
```json
{
	"action": "update(1)",
	"variables": {
        "current_index": 0,
        "target_waypoints": [1, 2, 3],
        "next_waypoint": 1
    },
	"global_trans": [],
    "initial_state": "ENGAGE_PERSON",

	"states": {
		"ENGAGE_PERSON": "walk to person, say hello -> PLAY",

		"PLAY": {
			"action": "do a fun dance or follow the person",
            "update": {
		        "current_index": "current_index + 1",
		        "next_waypoint": "target_waypoints[current_index] if current_index < len(target_waypoints) else None"
		    },
			"trans": [
				"DONE: current_index >= 3"
			]
		}
	}
}
```
"""
def test_s2_plan():
    S2DPlan.init_default()
    plan = S2DPlan("Play with the person for 3 minutes")
    print(S2DPlan.get_s2d_input())
    S2DPlan.process_s2d_response(test_response)
    print(plan.get_current_state())
    plan.transit("PLAY")
    plan.transit("PLAY")

    print(plan.get_current_state())
    print(plan.variables)

    plan.transit("DONE")
    # assert plan.get_current_state() == "DONE"
    # assert plan.status == STATUS_SUCCESS

    # plan = S2DPlan.parse("Play with the person for 2 minutes", test_response)
    # print(S2DPlan.get_history_sorted_by_time())
    # S2DPlan.init_default()
    print(S2DPlan.get_s2d_input())

# test_s2_plan()