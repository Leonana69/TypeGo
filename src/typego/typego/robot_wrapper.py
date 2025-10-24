from abc import ABC, abstractmethod
import re, time
import inspect

from typego.robot_info import RobotInfo
from typego.skill_item import SkillRegistry, SubSystem
import typego.frontend_message as frontend_message
from typego.robot_observation import RobotObservation

def robot_skill(name: str, description: str = "",
                subsystem: SubSystem = SubSystem.DEFAULT):
    """
    Decorator to register a robot skill.
    """
    def deco(fn):
        sig = inspect.signature(fn)
        params = {
            n: p.annotation
            for n, p in sig.parameters.items()
            if n != "self" and p.annotation != inspect._empty
        }
        setattr(fn, "_skill_name", name)
        setattr(fn, "_skill_description", description)
        setattr(fn, "_skill_params", params)
        setattr(fn, "_skill_subsystem", subsystem)
        # Record whether skill accepts control events explicitly
        fn.__accepts_stop__  = "stop_event"  in sig.parameters
        fn.__accepts_pause__ = "pause_event" in sig.parameters
        fn.__accepts_task_id__ = "task_id" in sig.parameters
        return fn
    return deco

class RobotWrapper(ABC):
    """
    Base class for robot wrappers.
    This class provides a common interface for different robot implementations.
    It manages robot information, observation, and skill registry."""
    def __init__(self, robot_info: RobotInfo, obs: RobotObservation):
        self.robot_info = robot_info
        self.obs = obs

        self.registry = SkillRegistry()
        self._auto_register_skills()

        self.running = False

    def _auto_register_skills(self):
        declared_skills: dict[str, tuple[str, str, dict, SubSystem]] = {}

        for cls in self.__class__.mro():
            for name, obj in cls.__dict__.items():
                func = obj
                if isinstance(func, (staticmethod, classmethod)):
                    func = func.__func__
                if not callable(func):
                    continue
                sk_name = getattr(func, "_skill_name", None)
                if not sk_name:
                    continue
                sk_desc = getattr(func, "_skill_description", "")
                sk_params = getattr(func, "_skill_params", {})
                sk_subsys = getattr(func, "_skill_subsystem", SubSystem.DEFAULT)
                declared_skills.setdefault(sk_name, (name, sk_desc, sk_params, sk_subsys))

        for skill_name, (method_name, sk_desc, sk_params, sk_subsys) in declared_skills.items():
            bound = getattr(self, method_name, None)
            if not callable(bound):
                continue

            self.registry.register(
                skill_name, description=sk_desc, params=sk_params, subsystem=sk_subsys
            )(bound)

    # ---- Public control APIs (global or per-subsystem) ----
    def pause_action(self) -> bool:
        ret = self.registry.pause()
        self._pause_action()
        return ret

    def resume_action(self) -> bool:
        ret = self.registry.resume()
        self._resume_action()
        return ret

    def stop_action(self) -> bool:
        ret = self.registry.stop()
        self._stop_action()
        return ret

    def start(self):
        if self.running:
            return
        self.running = True
        self.obs.start()
        self._start()

    def stop(self):
        if not self.running:
            return
        self.running = False
        self._stop()
        self.obs.stop()

    # ---- Abstract methods to be implemented specifically for each robot ----
    @abstractmethod
    def _start(self) -> bool:
        """Called once when starting the robot."""
        ...

    @abstractmethod
    def _stop(self):
        """Called once when stopping the robot."""
        ...

    @abstractmethod
    def _pause_action(self):
        """Call once when pausing the robot's current action."""
        ...

    @abstractmethod
    def _resume_action(self):
        """Call once when resuming the robot's current action."""
        ...

    @abstractmethod
    def _stop_action(self):
        """Call once when stopping the robot's current action."""
        ...

    # ---- Commons skills ----
    @robot_skill("take_picture", description="Take a picture and save it", subsystem=SubSystem.DEFAULT)
    def take_picture(self, task_id: int) -> bool:
        frontend_message.publish(self.obs.rgb_image, task_id)
        return True

    @robot_skill("log", description="Show user a message", subsystem=SubSystem.DEFAULT)
    def log(self, message: str, task_id: int) -> bool:
        frontend_message.publish(message, task_id)
        return True