from dataclasses import dataclass
from datetime import datetime
import json, time

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
    content: str
    status: str

    def __repr__(self) -> str:
        js = {
            "start": format_time(self.start),
            "end": format_time(self.end),
            "action": self.content,
            "status": self.status
        }
        return json.dumps(js, indent=4)
    
    def is_finished(self):
        """Mark the action as finished with a result."""
        return self.status in (STATUS_SUCCESS, STATUS_FAILED, STATUS_STOPPED)
    
    def finish(self, result: bool):
        """Finish the action with a result."""
        if self.status == STATUS_IN_PROGRESS:
            self.end = time.time()
            self.status = STATUS_SUCCESS if result else STATUS_FAILED

class S2Plan:
    ID_COUNTER = 0
    HISTORY: dict[int, "S2Plan"] = {}
    CURRENT: "S2Plan" = None
    def __init__(self, content: str, plan_json: dict):
        self.id = S2Plan.ID_COUNTER
        S2Plan.ID_COUNTER += 1

        self.local_data = plan_json.get("local_data", {})
        self.global_conditions = plan_json.get("global_conditions", [])
        self.states = plan_json["states"]
        self.current_state = "INIT"

        self.start_time = time.time()
        self.end_time = None
        self.content = content
        self.status = STATUS_IN_PROGRESS

        S2Plan.HISTORY[self.id] = self
        S2Plan.CURRENT = self

        after_init_states = self.states[self.current_state].get("transitions", [])
        for transition in after_init_states:
            if transition.endswith(": always"):
                self.transit(transition[:-8])
            else:
                raise ValueError(f"Invalid transition after initialization: {transition}")
    
    @classmethod
    def init_default(cls):
        """Initialize a default plan to track initial robot actions before any plan arrives."""
        default_plan = cls.__new__(cls)  # bypass __init__
        default_plan.id = cls.ID_COUNTER
        cls.ID_COUNTER += 1

        default_plan.local_data = {}
        default_plan.global_conditions = []
        default_plan.states = {"INIT": {"action": "Idle/Bootstrap", "transitions": []}}
        default_plan.current_state = "INIT"

        default_plan.start_time = time.time()
        default_plan.end_time = None
        default_plan.content = DEFAULT_PLAN_CONTENT
        default_plan.status = STATUS_IN_PROGRESS
        default_plan.action_list: list[ActionItem] = []  # type: ignore # track initial actions

        cls.HISTORY[default_plan.id] = default_plan
        cls.CURRENT = default_plan

    @classmethod
    def set_default(cls):
        cls.CURRENT = cls.HISTORY.get(0)
        cls.CURRENT.action_list = []  # reset action list for default plan

    @classmethod
    def parse(cls, instruct: str, llm_response: str):
        try:
            json_content = llm_response.split("```json", 1)[1].rsplit("```", 1)[0]
            plan_json = json.loads(json_content)
        except Exception as e:
            raise ValueError(f"Failed to parse plan: {e}")

        acts = plan_json.get("instruction_actions", "").split(";")
        for act in acts:
            act = act.strip()
            if not act:
                continue

            if act.startswith("stop("):
                task_id = int(act[5:-1].strip())
                plan = cls.HISTORY.get(task_id)
                if plan:
                    plan.status = STATUS_STOPPED
                    plan.end_time = time.time()
                    cls.CURRENT = None

            elif act.startswith("continue("):
                task_id = int(act[9:-1].strip())
                plan = cls.HISTORY.get(task_id)
                if plan:
                    plan.status = STATUS_IN_PROGRESS
                    plan.start_time = time.time()
                    plan.end_time = None
                    cls.CURRENT = plan

            elif act.startswith("pause("):
                task_id = int(act[6:-1].strip())
                plan = cls.HISTORY.get(task_id)
                if plan:
                    plan.status = STATUS_PAUSED
                    plan.end_time = time.time()
                    cls.CURRENT = None

            elif act.startswith("new("):
                return cls(instruct, plan_json)

            else:
                raise ValueError(f"Unknown action: {act}")

    @classmethod
    def get_history_sorted_by_time(cls, after_time: float = None):
        plans = cls.HISTORY.values()
        if after_time is not None:
            plans = filter(lambda p: p.start_time > after_time, plans)
        return sorted(plans, key=lambda p: p.start_time)

    def get_current_state(self):
        return self.current_state

    def transit(self, new_state: str):
        if new_state not in self.states:
            raise ValueError(f"Invalid state: {new_state}")
        
        if self.current_state == new_state:
            return
        self.action_list: list[ActionItem] = []
        self.current_state = new_state
        self.update_local_data()

        if new_state == "DONE":
            self.status = STATUS_SUCCESS
            self.end_time = time.time()
            S2Plan.CURRENT = None

    def process_s1_response(self, response: str) -> str:
        if response.startswith('```json'):
            response = response[7:-3].strip()
        json_response = json.loads(response)
        next_action = json_response.get("next_action", "continue()")
        next_state = json_response.get("transition", "null")
        if next_state and next_state != "null":
            self.transit(next_state)
        return next_action

    def add_action(self, action: str):
        """Add an action to the current state."""
        if self.status != STATUS_IN_PROGRESS:
            raise ValueError("Cannot add action to a non-in-progress plan.")
        
        action_item = ActionItem(start=time.time(), end=None, content=action, status=STATUS_IN_PROGRESS)
        self.action_list.append(action_item)

    def finish_action(self, result: bool):
        """Finish the last action with a result."""
        if not self.action_list or not any(action.status == STATUS_IN_PROGRESS for action in self.action_list):
            raise ValueError("Cannot finish action: no actions in progress.")

        for action in reversed(self.action_list):
            if action.status == STATUS_IN_PROGRESS:
                action.finish(result)
                break

    def update_local_data(self):
        updates = self.states[self.current_state].get("local_data_update", {})
        for key, expr in updates.items():
            try:
                self.local_data[key] = eval(expr, {}, self.local_data.copy())
            except Exception as e:
                print(f"Failed to evaluate local_data_update for key '{key}': {expr}")
                raise e
            
    def __repr__(self) -> str:
        return json.dumps({
            "id": str(self.id),
            "start": format_time(self.start_time),
            "end": format_time(self.end_time),
            "content": self.content,
            "status": self.status
        })
    
    def get_s1_input(self) -> str:
        if self.content == DEFAULT_PLAN_CONTENT:
            return "None"
        info = {
            "local_data": self.local_data,
            "global_conditions": self.global_conditions,
            "current_state": {
                self.states[self.current_state],
            }
        }
        return json.dumps(info, indent=4)

test_response = """
```json
{
	"instruction_actions": "new()",
	"local_data": {
        "current_index": 0,
        "target_waypoints": [1, 2, 3],
        "next_waypoint": 1
    },
	"global_conditions": [],

	"states": {
		"INIT": {
			"action": null,
			"transitions": ["ENGAGE_PERSON: always"]
		},

		"ENGAGE_PERSON": {
			"action": "walk to person, say hello",
			"transitions": ["PLAY: always"]
		},

		"PLAY": {
			"action": "do a fun dance or follow the person",
            "local_data_update": {
		        "current_index": "current_index + 1",
		        "next_waypoint": "target_waypoints[current_index] if current_index < len(target_waypoints) else None"
		    },
			"transitions": [
				"DONE: current_index >= 3"
			]
		},

		"DONE": {
			"action": null,
			"transitions": []
		}
	}
}
```
"""
def test_s2_plan():
    # plan = S2Plan.parse("Play with the person for 3 minutes", test_response)
    # print(plan.get_current_state())
    # plan.transit("PLAY")
    # print(plan.local_data)
    # plan.transit("PLAY")
    # print(plan.local_data)
    # plan.transit("PLAY")
    # print(plan.local_data)

    # plan.transit("DONE")
    # assert plan.get_current_state() == "DONE"
    # assert plan.status == STATUS_SUCCESS

    # plan = S2Plan.parse("Play with the person for 2 minutes", test_response)
    # print(S2Plan.get_history_sorted_by_time())
    S2Plan.init_default()
    print(S2Plan.get_history_sorted_by_time())

test_s2_plan()