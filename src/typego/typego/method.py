from dataclasses import dataclass, field
from typing import Any, Literal, Optional
import time, json

from typego.llm_wrapper import LLMWrapper
from typego.skill_item import SkillRegistry

# ------------------------------
# Method spec + frames
# ------------------------------
@dataclass
class MethodSpec:
    name: str
    description: str
    goal: dict[str, Any]
    obs_keys: list[str]
    api: list[str]                   # symbols the LLM may call (skill or sub-method names)
    termination: list[str]           # Python exprs over ctx {obs, goal, memory, state}
    submethods: dict[str, "MethodSpec"] = field(default_factory=dict)  # name -> MethodSpec
    budgets: dict[str, Any] = field(default_factory=lambda: {
        "max_steps": 200, "max_secs": 30.0
    })
    policy_hints: list[str] = field(default_factory=list)

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
    return (
        "# ROLE\n"
        f"You are executing method: {spec.name}\n\n"
        "# CONTEXT\n"
        f"Description: {spec.description}\n"
        f"Goal: {json.dumps(spec.goal)}\n"
        f"Obs: {json.dumps(obs)}\n"
        f"State: {json.dumps(state)}\n"
        f"API symbols you may call: {spec.api}\n\n"
        "# OUTPUT\n"
        "Return a JSON object with keys {\"call\", \"args\", \"confidence\"}. "
        "Valid 'call' is one of the API symbols above. 'args' is a JSON object."
    )

def default_validate_json_call(raw: str, allowed_calls: list[str]) -> Optional[dict[str, Any]]:
    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict): return None
        if obj.get("call") in allowed_calls and isinstance(obj.get("args", {}), dict):
            return {"call": obj["call"], "args": obj.get("args", {}), "confidence": obj.get("confidence", 0.0)}
    except Exception:
        return None
    return None

def eval_termination(pred: str, ctx: dict[str, Any]) -> bool:
    # Keep this tight: no builtins; only ctx names.
    return bool(eval(pred, {"__builtins__": {}}, ctx))

# ------------------------------
# Engine
# ------------------------------
class MethodEngine:
    """
    - Allows LLM to call either skills or sub-methods listed in spec.api.
    - Runs a call-stack of Frames, so sub-methods compose naturally.
    - Observations are requested per current frame via `observe_fn(obs_keys)`.
    """
    def __init__(self, skill_registry: SkillRegistry, prompt_fn=default_prompt, validate_fn=default_validate_json_call):
        self.llm = LLMWrapper()
        self.registry = skill_registry
        self.prompt_fn = prompt_fn
        self.validate_fn = validate_fn

    def run(self, root_spec: MethodSpec, observe_fn):
        trace: list[dict[str, Any]] = []
        state = {"global_steps": 0, "photos": 0}  # example aggregate; add what you need
        stack: list[Frame] = [Frame(
            spec=root_spec, goal=root_spec.goal, memory={}, step=0, t0=time.time(), name=root_spec.name
        )]

        while stack:
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
            obs = observe_fn(fr.spec.obs_keys)  # your observer should honor requested keys
            obs["t"] = time.time() - fr.t0

            # Frame termination?
            term_ctx = {"obs": obs, "goal": fr.goal, "memory": fr.memory, "state": state}
            if any(eval_termination(pred, term_ctx) for pred in fr.spec.termination):
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

            # Ask LLM what to do next (skills or sub-methods)
            prompt = self.prompt_fn(fr.spec, obs, {**state, **fr.memory})
            raw = self.llm.request(prompt)
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
                result = self.registry.exec(symbol, **args)
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
                # Voluntary stop of current frame; bubbles up
                popped = stack.pop()
                status = "ok" if args.get("ok", False) else "halted"
                trace.append({"event": "submethod_exit", "method": popped.name, "status": status, "raw": raw})
                if not stack:
                    return {"status": status, "trace": trace, "state": state}
            else:
                # Unknown symbol; log and continue to avoid deadlock
                trace.append({"event": "invalid_call", "method": fr.name, "symbol": symbol, "raw": raw})
                fr.step += 1
                state["global_steps"] += 1

def make_find_object_method() -> MethodSpec:
    return MethodSpec(
        name="find_object",
        description="Search environment until the target object is confidently detected in view.",
        goal={"object": "<name>", "min_conf": 0.8},
        obs_keys=["detections", "pose", "map", "camera_ready"],
        api=[  # skills only in this leaf method (you can add more)
            "orienting", "look_object", "move_left", "move_right",
            "move_forward", "move_back", "turn_left", "turn_right", "log"
        ],
        termination=[
            # True when detections contain our target with sufficient confidence
            "goal['object'] in obs.get('detections', {}) and obs['detections'][goal['object']]['conf'] >= goal.get('min_conf', 0.8)"
        ],
    )