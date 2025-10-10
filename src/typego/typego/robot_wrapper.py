from abc import ABC, abstractmethod
import re, time
import inspect

from typego.robot_info import RobotInfo
from typego.yolo_client import ObjectInfo
from typego.skill_item import SkillRegistry, SubSystem
from typego.frontend_message import publish
from typego.robot_observation import RobotObservation

# =========================
# Decorator to tag skills
# =========================
def robot_skill(name: str, description: str = "",
                subsystem: SubSystem = SubSystem.DEFAULT):
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
        return fn
    return deco

class RobotWrapper(ABC):
    def __init__(self, robot_info: RobotInfo, observation: RobotObservation):
        self.robot_info = robot_info
        self.observation = observation

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
            raise RuntimeError("Robot is already running")
        self.running = True
        self.observation.start()
        self._start()

    def stop(self):
        if not self.running:
            return
        self.running = False
        self._stop()
        self.observation.stop()

    @abstractmethod
    def _start(self) -> bool:
        """
        Start the robot.
        """
        ...

    @abstractmethod
    def _stop(self):
        """
        Stop the robot.
        """
        ...

    @abstractmethod
    def _pause_action(self):
        """
        Pause the robot's current action.
        """
        ...

    @abstractmethod
    def _resume_action(self):
        """
        Resume the robot's current action.
        """
        ...

    @abstractmethod
    def _stop_action(self):
        """
        Stop the robot's current action.
        """
        ...

    @robot_skill("take_picture", description="Take a picture and save it", subsystem=SubSystem.DEFAULT)
    def take_picture(self) -> bool:
        publish(self.observation.image)
        return True

    @robot_skill("log", description="Log a message", subsystem=SubSystem.DEFAULT)
    def log(self, message: str) -> bool:
        publish(message)
        return True

    # vision skills
    def get_obj_list(self) -> list[ObjectInfo]:
        """Returns a formatted string of detected objects."""
        process_result = self.observation.fetch_processed_result()
        return process_result[1] if process_result else []

    def get_obj_list_str(self) -> str:
        """Returns a formatted string of detected objects."""
        object_list = self.get_obj_list()
        return f"[{', '.join(str(obj) for obj in object_list)}]"

    def get_obj_info(self, object_name: str, reliable=False) -> ObjectInfo | None:
        object_name = object_name.strip('\'').strip('"').lower()

        for _ in range(3):
            object_list = self.get_obj_list()
            for obj in object_list:
                if obj.name.startswith(object_name):
                    return obj
            if not reliable:
                break
            time.sleep(0.2)
        return None

    def is_visible(self, object_name: str, reliable=False) -> bool:
        return self.get_obj_info(object_name, reliable) is not None

    def _get_object_attribute(self, object_name: str, attr: str) -> float | str:
        """Helper function to retrieve an object's attribute."""
        info = self.get_obj_info(object_name)
        if info is None:
            return f'{attr}: {object_name} is not in sight'
        return getattr(info, attr)
    
    def object_x(self, object_name: str) -> float | str:
        # if `[float]` is in the object_name, use it
        match = re.search(r'\[(-?\d+(\.\d+)?)\]', object_name)
        if match:
            # Extract the number and return it as a float
            extracted_number = float(match.group(1))
            return extracted_number
        return self._get_object_attribute(object_name, 'x')
    
    def object_y(self, object_name: str) -> float | str:
        return self._get_object_attribute(object_name, 'y')
    
    def object_width(self, object_name: str) -> float | str:
        return self._get_object_attribute(object_name, 'w')
    
    def object_height(self, object_name: str) -> float | str:
        return self._get_object_attribute(object_name, 'h')
    
    def object_distance(self, object_name: str) -> float | str:
        return self._get_object_attribute(object_name, 'depth')