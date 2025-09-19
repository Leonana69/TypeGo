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
    state: str
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

class S2DPlan:
    ID_COUNTER = 0
    HISTORY: dict[int, "S2DPlan"] = {}
    CURRENT: "S2DPlan" = None
    def __init__(self, content: str, plan_json: dict):
        self.id = S2DPlan.ID_COUNTER
        S2DPlan.ID_COUNTER += 1

        self.local_data = plan_json.get("local_data", {})
        self.global_trans = plan_json.get("global_trans", [])
        self.states = plan_json["states"]
        self.current_state = "INIT"

        self.start_time = time.time()
        self.end_time = None
        self.content = content
        self.status = STATUS_IN_PROGRESS

        self.s2s_history: list[ActionItem] = []  # track actions in this plan

        S2DPlan.HISTORY[self.id] = self
        S2DPlan.CURRENT = self

        after_init_states = self.states[self.current_state].get("trans", [])
        for transition in after_init_states:
            if transition.endswith(": always"):
                self.transit(transition[:-8])
            else:
                raise ValueError(f"Invalid transition after initialization: {transition}")
    
    @classmethod
    def init_default(cls):
        """Initialize a default plan to track initial robot actions before any plan arrives."""
        default_plan = cls.__new__(cls)  # bypass __init__
        default_plan.id = 0
        cls.ID_COUNTER = 1

        default_plan.local_data = {}
        default_plan.global_trans = [
            "CHASE_BALL: see a sports ball"
        ]
        default_plan.states = {
            "IDLE": {
                "action": None,
                "trans": []
            },
            "CHASE_BALL": {
                "action": "follow the sports ball",
                "trans": ["IDLE: always"]
            }
        }
        default_plan.current_state = "IDLE"

        default_plan.start_time = time.time()
        default_plan.end_time = None
        default_plan.content = DEFAULT_PLAN_CONTENT
        default_plan.status = STATUS_IN_PROGRESS

        cls.HISTORY[default_plan.id] = default_plan
        cls.CURRENT = default_plan

    @classmethod
    def set_default(cls):
        cls.CURRENT = cls.HISTORY.get(0)
        cls.CURRENT.status = STATUS_IN_PROGRESS
        cls.CURRENT.start_time = time.time()
        cls.CURRENT.end_time = None
        cls.CURRENT.s2s_history = []  # reset action list for default plan

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
        
        self.current_state = new_state
        self.update_local_data()

        if new_state == "DONE":
            self.status = STATUS_SUCCESS
            self.end_time = time.time()
            S2DPlan.CURRENT = None

    def process_s1_response(self, response: str) -> str | None:
        if response.startswith('```json'):
            response = response[7:-3].strip()

        json_response = json.loads(response)
        next_action = json_response.get("action", None)
        next_state = json_response.get("trans", None)
        if next_state:
            self.transit(next_state)

        return next_action

    def add_action(self, action: str):
        """Add an action to the current state."""
        if self.status != STATUS_IN_PROGRESS:
            raise ValueError("Cannot add action to a non-in-progress plan.")

        action_item = ActionItem(start=time.time(), end=None, state=self.current_state, content=action, status=STATUS_IN_PROGRESS)
        self.s2s_history.append(action_item)

    def finish_action(self, result: bool):
        """Finish the last action with a result."""
        # TODO: fix this function
        if not self.s2s_history or not any(action.status == STATUS_IN_PROGRESS for action in self.s2s_history):
            raise ValueError("Cannot finish action: no actions in progress.")

        for action in reversed(self.s2s_history):
            if action.status == STATUS_IN_PROGRESS:
                action.finish(result)
                break

    def update_local_data(self):
        updates = self.states[self.current_state].get("update", {})
        for key, expr in updates.items():
            try:
                self.local_data[key] = eval(expr, {}, self.local_data.copy())
            except Exception as e:
                print(f"Failed to evaluate update for key '{key}': {expr}")
                raise e
            
    def __repr__(self) -> str:
        return json.dumps({
            "id": str(self.id),
            "start": format_time(self.start_time),
            "end": format_time(self.end_time),
            "content": self.content,
            "status": self.status
        })
    
    def get_s2s_input(self) -> str:
        info = {
            "local_data": self.local_data,
            "global_trans": self.global_trans,
            "current_state": {
                self.current_state: self.states[self.current_state],
            },

            "action_history": [str(action) for action in self.s2s_history if action.is_finished()]
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
	"global_trans": [],

	"states": {
		"INIT": {
			"trans": ["ENGAGE_PERSON: always"]
		},

		"ENGAGE_PERSON": {
			"action": "walk to person, say hello",
			"trans": ["PLAY: always"]
		},

		"PLAY": {
			"action": "do a fun dance or follow the person",
            "update": {
		        "current_index": "current_index + 1",
		        "next_waypoint": "target_waypoints[current_index] if current_index < len(target_waypoints) else None"
		    },
			"trans": [
				"DONE: current_index >= 3"
			]
		},

		"DONE": {}
	}
}
```
"""
def test_s2_plan():
    # plan = S2DPlan.parse("Play with the person for 3 minutes", test_response)
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

    # plan = S2DPlan.parse("Play with the person for 2 minutes", test_response)
    # print(S2DPlan.get_history_sorted_by_time())
    S2DPlan.init_default()
    print(S2DPlan.get_history_sorted_by_time())

# test_s2_plan()