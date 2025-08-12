from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from numpy import ndarray
import time, threading
from PIL import Image
import asyncio, cv2
import re
import numpy as np
from collections import deque
from enum import Enum
import inspect

from typego.robot_info import RobotInfo
from typego.yolo_client import ObjectInfo
from typego.skill_item import SKILL_RET_TYPE, SkillRegistry
from typego.utils import evaluate_value

from typego_interface.msg import WayPointArray, WayPoint

class RobotPosture(Enum):
    STANDING = "standing"
    LYING = "lying"
    MOVING = "moving"

    @staticmethod
    def from_string(s: str):
        if s == "standing":
            return RobotPosture.STANDING
        elif s == "lying":
            return RobotPosture.LYING
        elif s == "moving":
            return RobotPosture.MOVING
        else:
            raise ValueError(f"Unknown posture: {s}")

class SLAMMap:
    def __init__(self):
        self.map_data: Optional[ndarray] = None
        self.width: int = 0
        self.height: int = 0
        self.origin: tuple[float, float] = (0.0, 0.0)
        self.resolution: float = 0.0

        self.robot_loc: tuple[float, float] = (0.0, 0.0)
        self.robot_yaw: float = 0.0

        self.waypoints: Optional[WayPointArray] = None

        # Stores (timestamp, (x, y)) tuples for up to 60 seconds
        self.trajectory = deque()  # type: deque[tuple[float, tuple[float, float]]]

    def update_map(self, map_data: ndarray, width: int, height: int, origin: tuple[float, float], resolution: float):
        self.map_data = map_data
        self.width = width
        self.height = height
        self.origin = origin
        self.resolution = resolution

    def update_waypoints(self, waypoints: WayPointArray):
        self.waypoints = waypoints

    def update_robot_state(self, robot_loc: tuple[float, float], robot_yaw: float):
        self.robot_loc = robot_loc
        self.robot_yaw = robot_yaw
        now = time.time()

        self.trajectory.append((now, robot_loc))
        # Remove points older than 60 seconds
        while self.trajectory and now - self.trajectory[0][0] > 60:
            self.trajectory.popleft()

    def get_waypoint(self, id: int) -> Optional[WayPoint]:
        if self.waypoints is None:
            return None

        for waypoint in self.waypoints.waypoints:
            if waypoint.id == id:
                return waypoint
        return None
    
    def get_nearest_waypoint_id(self, loc: tuple[float, float]) -> int | None:
        if self.waypoints is None or not self.waypoints.waypoints:
            return None

        min_dist = float('inf')
        nearest_id = None
        for waypoint in self.waypoints.waypoints:
            distance = np.sqrt((waypoint.x - loc[0]) ** 2 + (waypoint.y - loc[1]) ** 2)
            if distance < min_dist:
                min_dist = distance
                nearest_id = waypoint.id
        return nearest_id

    def get_waypoint_list_str(self) -> str:
        if self.waypoints is None:
            return "[]\n"

        waypoint_list = "[\n"
        for waypoint in self.waypoints.waypoints:
            # waypoint_list += (f"    {{\"id\": {waypoint.id}, \"loc\": [{round(waypoint.x, 2)}, {round(waypoint.y, 2)}], \"label\": \"{waypoint.label}\"}},\n")
            waypoint_list += (f"    {{\"id\": {waypoint.id}, \"label\": \"{waypoint.label}\"}},\n")
        waypoint_list += "]"
        return waypoint_list

    def get_map(self) -> Optional[ndarray]:
        if self.map_data is None:
            return None

        u = int((self.robot_loc[0] - self.origin[0]) / self.resolution)
        v = self.height - int((self.robot_loc[1] - self.origin[1]) / self.resolution)

        map_image = np.zeros((self.height, self.width), dtype=np.uint8)
        map_image[self.map_data == 0] = 255
        map_image[self.map_data == -1] = 127
        map_image[self.map_data > 0] = 0
        map_image = cv2.flip(map_image, 0)
        map_image = cv2.cvtColor(map_image, cv2.COLOR_GRAY2BGR)

        # Draw trajectory
        # now = time.time()
        # trajectory_copy = list(self.trajectory)
        # for i in range(1, len(trajectory_copy)):
        #     t0, p0 = trajectory_copy[i - 1]
        #     t1, p1 = trajectory_copy[i]
        #     age = (now - t0) / 60.0  # normalized age [0, 1]
        #     age = min(max(age, 0.0), 1.0)

        #     # Color fades from green (0,255,0) to red (0,0,255)
        #     color = (
        #         int(0),                 # Blue
        #         int(255 * (1 - age)),   # Green
        #         int(255 * age)          # Red
        #     )

        #     u0 = int((p0[0] - self.origin[0]) / self.resolution)
        #     v0 = self.height - int((p0[1] - self.origin[1]) / self.resolution)
        #     u1 = int((p1[0] - self.origin[0]) / self.resolution)
        #     v1 = self.height - int((p1[1] - self.origin[1]) / self.resolution)
        #     cv2.line(map_image, (u0, v0), (u1, v1), color, 2)

        # Draw robot position and orientation
        cv2.circle(map_image, (u, v), 3, (0, 255, 0), -1)
        cv2.arrowedLine(map_image, (u, v), (int(u + 10 * np.cos(self.robot_yaw)), int(v - 10 * np.sin(self.robot_yaw))), (0, 255, 0), 1)

        # Draw waypoints
        if self.waypoints is not None:
            for waypoint in self.waypoints.waypoints:
                u = int((waypoint.x - self.origin[0]) / self.resolution)
                v = self.height - int((waypoint.y - self.origin[1]) / self.resolution)
                cv2.circle(map_image, (u, v), 3, (255, 0, 0), -1)
                cv2.putText(map_image, f"{waypoint.id}", (u + 3, v), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        map_image = cv2.resize(map_image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
        return map_image
    
    def is_empty(self) -> bool:
        return self.map_data is None or self.map_data.size == 0

class RobotObservation(ABC):
    def __init__(self, robot_info: RobotInfo, rate: int):
        self.interval: float = 1.0 / rate
        self.robot_info = robot_info

        self._image: Optional[Image.Image] = None
        self._depth: Optional[ndarray] = None
        self._orientation: ndarray = np.array([0.0, 0.0, 0.0])
        self._position: ndarray = np.array([0.0, 0.0, 0.0])
        self._slam_map: SLAMMap = SLAMMap()

        # Add individual locks for each property
        self._image_lock = threading.Lock()
        self._depth_lock = threading.Lock()
        self._orientation_lock = threading.Lock()
        self._position_lock = threading.Lock()
        self._slam_map_lock = threading.Lock()

        self.running: bool = False
        self.thread = threading.Thread(target=self.update_observation, daemon=True)

    def start(self):
        self.running = True
        self._start()
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()
        self._stop()

    @abstractmethod
    def _start(self):
        pass

    @abstractmethod
    def _stop(self):
        pass

    @property
    def image(self) -> Optional[Image.Image]:
        with self._image_lock:
            return self._image
    
    @image.setter
    def image(self, value: Image.Image):
        with self._image_lock:
            self._image = value

    @property
    def depth(self) -> Optional[ndarray]:
        with self._depth_lock:
            return self._depth
        
    @depth.setter
    def depth(self, value: ndarray):
        with self._depth_lock:
            self._depth = value

    @property
    def orientation(self) -> ndarray:
        with self._orientation_lock:
            return self._orientation
        
    @orientation.setter
    def orientation(self, value: ndarray):
        with self._orientation_lock:
            self._orientation = value

    @property
    def position(self) -> ndarray:
        with self._position_lock:
            return self._position
        
    @position.setter
    def position(self, value: ndarray):
        with self._position_lock:
            self._position = value
        
    @property
    def slam_map(self) -> SLAMMap:
        with self._slam_map_lock:
            return self._slam_map
    
    @slam_map.setter
    def slam_map(self, value: SLAMMap):
        with self._slam_map_lock:
            self._slam_map = value
    
    def update_observation(self):
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def schedule_tasks():
            tasks: set[asyncio.Task] = set()
            
            while self.running:
                start_time = time.time()

                # Add a new task to the set
                if self._image is not None:
                    task = asyncio.create_task(self.process_image(self._image))
                    tasks.add(task)
                
                # Clean up completed tasks
                tasks = {t for t in tasks if not t.done()}
                # Sleep for the interval
                elapsed_time = time.time() - start_time
                await asyncio.sleep(max(0, self.interval - elapsed_time))
        # Run the async function in the event loop
        loop.run_until_complete(schedule_tasks())

    @abstractmethod
    async def process_image(self, image: Image.Image):
        pass
    
    @abstractmethod
    def fetch_processed_result(self) -> tuple[Image.Image, list] | None:
        pass

    def blocked(self) -> bool:
        return False
    
    def fetch_command(self) -> str | None:
        return None

def robot_skill(name: str, description: str = ""):
    def deco(fn):
        sig = inspect.signature(fn)
        params = {
            name: param.annotation
            for name, param in sig.parameters.items()
            if name != "self" and param.annotation != inspect._empty
        }
        setattr(fn, "_skill_name", name)
        setattr(fn, "_skill_description", description)
        setattr(fn, "_skill_params", params)
        return fn
    return deco

class RobotWrapper(ABC):
    def __init__(self, robot_info: RobotInfo, observation: RobotObservation, controller_func: dict[str, callable] = None):
        self.robot_info = robot_info
        self._observation = observation
        self._user_log = controller_func.get("user_log", lambda x: True)

        self.registry = SkillRegistry()
        self._auto_register_skills()

    def _auto_register_skills(self):
        """
        Find methods tagged with @skill on any class in the MRO, then register
        the *bound override* from this instance. Subclasses need not re-decorate.
        """
        # Map skill name -> (method_name_on_class, description)
        declared_skills: dict[str, tuple[str, str]] = {}

        for cls in self.__class__.mro():
            for name, obj in cls.__dict__.items():
                # unwrap abstractmethod / function wrappers to reach original
                func = obj
                if isinstance(func, (staticmethod, classmethod)):
                    func = func.__func__
                # only plain callables
                if not callable(func):
                    continue
                sk_name = getattr(func, "_skill_name", None)
                if not sk_name:
                    continue
                sk_desc = getattr(func, "_skill_description", "")
                sk_params = getattr(func, "_skill_params", {})
                # first occurrence in MRO wins (nearest subclass)
                declared_skills.setdefault(sk_name, (name, sk_desc, sk_params))

        for skill_name, (method_name, sk_desc, sk_params) in declared_skills.items():
            # get the *bound* method from the instance (subclass override)
            bound = getattr(self, method_name, None)
            if not callable(bound):
                continue
            # register; SkillRegistry will extract signature (ignores self)
            self.registry.register(skill_name, description=sk_desc, params=sk_params)(bound)

    @abstractmethod
    def start(self) -> bool:
        pass

    @abstractmethod
    def stop(self):
        pass

    @property
    def observation(self) -> RobotObservation: return self._observation

    # movement skills
    @abstractmethod
    @robot_skill("move", description="Move by (dx, dy) m distance (dx: +forward, dy: +left)")
    def move(self, dx: float, dy: float) -> bool:
        pass

    @abstractmethod
    @robot_skill("rotate", description="Rotate by deg degrees (deg: +left or clockwise)")
    def rotate(self, deg: float) -> bool:
        pass

    @abstractmethod
    def get_state(self) -> str:
        pass

    # vision skills
    def get_obj_list(self) -> list[ObjectInfo]:
        """Returns a formatted string of detected objects."""
        process_result = self._observation.fetch_processed_result()
        return process_result[1] if process_result else []

    def get_obj_list_str(self) -> str:
        """Returns a formatted string of detected objects."""
        object_list = self.get_obj_list()
        return f"[{', '.join(str(obj) for obj in object_list)}]"

    def get_obj_info(self, object_name: str) -> ObjectInfo | None:
        object_name = object_name.strip('\'').strip('"').lower()

        for _ in range(3):
            object_list = self.get_obj_list()
            for obj in object_list:
                if obj.name.startswith(object_name):
                    return obj
            time.sleep(0.2)
        return None

    def is_visible(self, object_name: str) -> bool:
        return self.get_obj_info(object_name) is not None

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

    @robot_skill("take_picture", description="Take a picture and save it")
    def take_picture(self) -> bool:
        return self._user_log(self.observation.image)

    @robot_skill("log", description="Log a message")
    def log(self, message: str) -> bool:
        return self._user_log(message)