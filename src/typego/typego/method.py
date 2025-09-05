from dataclasses import dataclass, field
from typing import Any, Literal, Optional
import time, json

from typego.llm_wrapper import LLMWrapper
from typego.skill_item import SkillRegistry
from typego.robot_wrapper import RobotWrapper
from typego.yolo_client import ObservationEncoder
from typego.utils import print_t

# ------------------------------
# Method spec + frames
# ------------------------------
@dataclass
class MethodSpec:
    name: str
    description: str
    robot: RobotWrapper
    goal: dict[str, Any]
    obs_keys: list[str]
    api: list[str]                   # symbols the LLM may call (skill or sub-method names)
    termination: list[str]           # Python exprs over ctx {obs, goal, memory, state}
    submethods: dict[str, "MethodSpec"] = field(default_factory=dict)  # name -> MethodSpec
    budgets: dict[str, Any] = field(default_factory=lambda: {
        "max_steps": 200, "max_secs": 30.0
    })
    policy_hints: list[str] = field(default_factory=list)
    logic: str | None = None

    def bind(self, **goal_overrides) -> "MethodSpec":
        # shallow copy + goal override
        m = MethodSpec(**{**self.__dict__})
        m.goal = {**self.goal, **goal_overrides}
        return m

@dataclass
class Frame:
    spec: MethodSpec
    goal: dict[str, Any]
    memory: dict[str, Any]
    step: int
    t0: float
    name: str

# ------------------------------
# Helpers (replace with your own if you have them)
# ------------------------------
def default_prompt(spec: MethodSpec, obs: dict[str, Any], state: dict[str, Any]) -> str:
    # Minimal but effective; keep your own richer template if you already have one.
    skills = spec.robot.registry.get_skill_list(keys=spec.api)
    return (
        "# ROLE\n"
        f"You are robot planner that executes a method called: {spec.name}.\n\n"
        "# CONTEXT\n"
        f"Description: {spec.description}\n"
        f"Goal: {json.dumps(spec.goal)}\n"
        f"Observation: {json.dumps(obs, cls=ObservationEncoder)}\n"
        f"State: {json.dumps(state)}\n"
        f"Available robot skills:\n{skills}\n\n"
        f"# POLICY HINTS\n{spec.policy_hints}\n\n"
        "# OUTPUT\n"
        "Return a JSON object with keys {\"call\", \"args\"}. "
        "Valid 'call' is one of the API symbols above. 'args' is a JSON object."
    )

def eval_termination(pred: str, ctx: dict[str, Any]) -> bool:
    safe_globals = {"__builtins__": {}}
    safe_globals.update(ctx)
    return bool(eval(pred, safe_globals, {}))

# ------------------------------
# Engine
# ------------------------------
class MethodEngine:
    """
    - Allows LLM to call either skills or sub-methods listed in spec.api.
    - Runs a call-stack of Frames, so sub-methods compose naturally.
    """
    def __init__(self, spec: MethodSpec, prompt_fn=default_prompt):
        self.llm = LLMWrapper()
        self.prompt_fn = prompt_fn
        self.spec = spec
        self.robot = spec.robot
        self.registry = spec.robot.registry

    def validate_fn(self, raw: str, allowed_calls: list[str]) -> Optional[dict[str, Any]]:
        if raw.startswith("```json") and raw.endswith("```"):
            raw = raw[8:-3].strip()
        try:
            obj = json.loads(raw)
            if not isinstance(obj, dict): return None
            if obj.get("call") in allowed_calls and isinstance(obj.get("args", {}), dict):
                return {"call": obj["call"], "args": obj.get("args", {})}
        except Exception:
            return None
        return None

    def run(self):
        trace: list[dict[str, Any]] = []
        state = {"global_steps": 0}  # example aggregate; add what you need
        stack: list[Frame] = [Frame(
            spec=self.spec, goal=self.spec.goal, memory={}, step=0, t0=time.time(), name=self.spec.name
        )]
        start_time = time.time()

        while stack:
            print(f"t: {time.time():.1f} | stack: {[f.name for f in stack]} | state: {state}")
            fr = stack[-1]
            # Budgets
            if fr.step >= fr.spec.budgets["max_steps"]:
                popped = stack.pop()
                parent_obs = {"last_call": f"method:{popped.name}", "last_result": "err:max_steps"}
                if stack:
                    # bubble an error up as a "submethod failed" event
                    trace.append({"event": "submethod_exit", "method": popped.name, "status": "max_steps"})
                    continue
                else:
                    return {"status": "max_steps", "trace": trace, "state": state}

            if (time.time() - fr.t0) >= fr.spec.budgets["max_secs"]:
                popped = stack.pop()
                if stack:
                    trace.append({"event": "submethod_exit", "method": popped.name, "status": "max_secs"})
                    continue
                else:
                    return {"status": "max_secs", "trace": trace, "state": state}

            # Observe for current frame
            full_obs = self.robot.observation.obs()  # your observer should honor requested keys
            print(full_obs)
            obs = {k: full_obs.get(k) for k in fr.spec.obs_keys}
            obs["t"] = int(time.time() - start_time)  # add elapsed time

            # Frame termination?
            term_ctx = {"obs": obs, "goal": fr.goal, "memory": fr.memory, "state": state}
            if any(eval_termination(pred, term_ctx) for pred in fr.spec.termination):
                print("[Method] Current method done.")
                popped = stack.pop()
                # Mark a submethod completion for parent (if any)
                if stack:
                    parent_obs = {"last_call": f"method:{popped.name}", "last_result": "ok"}
                    trace.append({"event": "submethod_exit", "method": popped.name, "status": "ok", "obs": obs})
                    # Parent can read child's outputs from popped.memory (if child wrote there)
                    stack[-1].memory[popped.name] = {"final_obs": obs, "memory": popped.memory}
                    continue
                else:
                    return {"status": "done", "trace": trace, "final_obs": obs, "state": state}

            if fr.spec.logic:
                # Execute custom logic if provided
                local_ctx = {"obs": obs, "goal": fr.goal, "memory": fr.memory, "state": state}
                try:
                    safe_globals = {"__builtins__": {}}
                    safe_globals.update(local_ctx)
                    raw = eval(fr.spec.logic, safe_globals, {})
                except Exception as e:
                    print(f"[Method] Error executing logic: {e}")
                    raw = None
            else:
                # Ask LLM what to do next (skills or sub-methods)
                prompt = self.prompt_fn(fr.spec, obs, {**state, **fr.memory})
                print_t(prompt)
                raw = self.llm.request(prompt)

            print_t(f"Decision raw: {raw}")
            call = self.validate_fn(raw, allowed_calls=fr.spec.api)

            if not call:
                # fallback; you can tailor this
                call = {"call": "halt", "args": {}, "confidence": 0.0, "why": "validation_failed"}

            symbol = call["call"]
            args = call.get("args", {})

            # Dispatch
            if symbol in fr.spec.submethods:
                # Treat args as goal overrides for the child method
                child_spec = fr.spec.submethods[symbol].bind(**args)
                stack.append(Frame(
                    spec=child_spec, goal=child_spec.goal, memory={}, step=0, t0=time.time(), name=child_spec.name
                ))
                trace.append({
                    "event": "submethod_enter", "parent": fr.name, "call": symbol, "goal": child_spec.goal, "raw": raw
                })
            elif symbol in self.registry.names():  # your Skills wrapper should expose available names
                print_t(f"[Method] Executing skill: {symbol} with args: {args}")
                result = self.registry.execute(symbol, args)
                print_t(f"[Method] Skill result: {result}")
                # Simple example side-effect: count successful photos
                if symbol == "take_picture" and result.get("ok"):
                    state["photos"] = state.get("photos", 0) + 1

                obs["last_call"] = symbol
                obs["last_result"] = "ok" if result.get("ok") else f"err:{result.get('error')}"

                trace.append({"event": "skill", "method": fr.name, "step": fr.step,
                              "obs": obs, "call": {"name": symbol, "args": args}, "result": result, "raw": raw})
                fr.memory["last_result"] = result
                fr.memory["last_call"] = symbol
                fr.step += 1
                state["global_steps"] += 1
            elif symbol == "halt":
                print("[Method] halt")
                # Voluntary stop of current frame; bubbles up
                popped = stack.pop()
                status = "ok" if args.get("ok", False) else "halted"
                trace.append({"event": "submethod_exit", "method": popped.name, "status": status, "raw": raw})
                if not stack:
                    return {"status": status, "trace": trace, "state": state}
            else:
                print(f"[Method] Unknown symbol: {symbol}")
                # Unknown symbol; log and continue to avoid deadlock
                trace.append({"event": "invalid_call", "method": fr.name, "symbol": symbol, "raw": raw})
                fr.step += 1
                state["global_steps"] += 1

def make_find_object_method(robot: RobotWrapper) -> MethodSpec:
    return MethodSpec(
        name="find_object",
        description="Search environment until the target object is confidently detected in view.",
        robot=robot,
        goal={"object": "<name>", "max_dist": 2.0},
        obs_keys=["t", "robot", "perception"],
        api=[  # skills only in this leaf method (you can add more)
            "orienting", "move_forward", "move_back", "turn_left", "turn_right", "log"
        ],
        termination=[
            # True when target object is found in perception list with sufficient confidence
            "True in [d['name'] == goal['object'] and d.get('dist', 0.0) <= goal.get('max_dist') for d in obs.get('perception')]"
        ],
        policy_hints=[
            "Use 'orienting' to align with the target object.",
            "Use 'move_forward' to approach the target object.",
            "If the target object is not in sight, use 'turn_left' or 'turn_right' with 30-degree steps to search the area."
        ]
    )

# def make_follow_object_method(robot: RobotWrapper) -> MethodSpec:
#     return MethodSpec(
#         name="follow_object",
#         description="Follow the target object until timeout.",
#         robot=robot,
#         goal={"object": "<name>", "max_dist": 1.5, "duration": 60.0},
#         obs_keys=["t", "robot", "perception"],
#         api=[  # skills only in this leaf method (you can add more)
#             "nav", "search"
#         ],
#         termination=[
#             # True when target object is found in perception list with sufficient confidence
#             "obs.get('t') >= goal.get('duration')"
#         ],
#         policy_hints=[
#             "Default to **`nav`** when the target is visible.",
#             "Use `vyaw` in `nav` to rotate toward the target's horizontal offset: If the object is to the right, use **negative vyaw**. If the object is to the left, use **positive vyaw**. Adjust until the target is centered (x location is around 0.5).",
#             "Use `vx` to maintain distance: Move forward if `dist > goal['max_dist']`. Stop or move backward if `dist <= goal['max_dist']`.",
#             "If the target object is not in sight, use 'search'."
#         ]
#     )

def make_follow_object_method(robot: RobotWrapper) -> MethodSpec:
    return MethodSpec(
        name="follow_object",
        description="Follow the target object until timeout.",
        robot=robot,
        goal={"object": "<name>", "max_dist": 1.5, "duration": 60.0},
        obs_keys=["t", "robot", "perception"],
        api=[  # skills only in this leaf method (you can add more)
            "follow"
        ],
        termination=[
            # True when target object is found in perception list with sufficient confidence
            "obs.get('t') >= goal.get('duration')"
        ],
        policy_hints=[
            "Use **`follow`** to track and follow the target object."
        ]
    )
