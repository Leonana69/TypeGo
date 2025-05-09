from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from numpy import ndarray
import time, threading
from PIL import Image
import asyncio, cv2
import re
import numpy as np

from typego.skillset import SkillSet
from typego.robot_info import RobotInfo
from typego.yolo_client import ObjectInfo
from typego.skill_item import SKILL_RET_TYPE
from typego.utils import evaluate_value

@dataclass
class MemoryItem:
    start: float
    end: float | None
    scene: str
    action: str

class RobotMemory:
    """
    'exec_history' is a list that stores the history of robot actions.
    """
    def __init__(self, robot_info: RobotInfo):
        self.robot_info = robot_info
        self.data: list[MemoryItem] = []

    def add(self, scene: str, action: str):
        if self.data:
            self.data[-1].end = time.time()
        self.data.append(MemoryItem(time.time(), None, scene, action))

    def get(self) -> str:
        current_time = time.time()
        self.data = [item for item in self.data if current_time - item.start < 60]
        rslt = ""
        for item in self.data:
            start_time = time.strftime("%H:%M:%S", time.localtime(item.start))
            if item.end is None:
                end_time = ""
            else:
                end_time = time.strftime("%H:%M:%S", time.localtime(item.end))
            rslt += f"[{start_time}~{end_time}]: {item.scene} -> {item.action}\n"

        return rslt

class SLAMMap:
    def __init__(self):
        self.map_data: Optional[ndarray] = None
        self.width: int = 0
        self.height: int = 0
        self.origin: tuple[float, float] = (0.0, 0.0)
        self.resolution: float = 0.0

        self.robot_loc: tuple[float, float] = (0.0, 0.0)
        self.robot_yaw: float = 0.0

    def update_map(self, map_data: ndarray, width: int, height: int, origin: tuple[float, float], resolution: float):
        self.map_data = map_data
        self.width = width
        self.height = height
        self.origin = origin
        self.resolution = resolution

    def update_robot_state(self, robot_loc: tuple[float, float], robot_yaw: float):
        self.robot_loc = robot_loc
        self.robot_yaw = robot_yaw

    def get_map(self) -> Optional[ndarray]:
        u = int((self.robot_loc[0] - self.origin[0]) / self.resolution)
        v = self.height - int((self.robot_loc[1] - self.origin[1]) / self.resolution)

        map_image = self.map_data.copy()
        cv2.circle(map_image, (u, v), 3, (0, 255, 0), -1)
        cv2.arrowedLine(map_image, (u, v), (int(u + 10 * np.cos(self.robot_yaw)), int(v - 10 * np.sin(self.robot_yaw))), (255, 0, 0), 1)
        return map_image

class RobotObservation(ABC):
    def __init__(self, robot_info: RobotInfo, rate: int):
        self.interval: float = 1.0 / rate
        self.robot_info = robot_info

        self._image: Optional[Image.Image] = None
        self._depth: Optional[ndarray] = None
        self._orientation: Optional[ndarray] = None
        self._position: Optional[ndarray] = None
        self._slam_map: SLAMMap = SLAMMap()

        # Add individual locks for each property
        self._image_lock = threading.Lock()
        self._depth_lock = threading.Lock()
        self._orientation_lock = threading.Lock()
        self._position_lock = threading.Lock()
        self._slam_map_lock = threading.Lock()

        self._image_process_result: tuple[Image.Image, list[ObjectInfo]] = (None, [])
        self._image_process_lock = threading.Lock()

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
    def orientation(self) -> Optional[ndarray]:
        with self._orientation_lock:
            return self._orientation
        
    @orientation.setter
    def orientation(self, value: ndarray):
        with self._orientation_lock:
            self._orientation = value

    @property
    def position(self) -> Optional[ndarray]:
        with self._position_lock:
            return self._position
        
    @position.setter
    def position(self, value: ndarray):
        with self._position_lock:
            self._position = value
    
    @property
    def image_process_result(self) -> tuple[Image.Image, list[ObjectInfo]]:
        with self._image_process_lock:
            return self._image_process_result
        
    @image_process_result.setter
    def image_process_result(self, value: tuple[Image.Image, list[ObjectInfo]]):
        with self._image_process_lock:
            self._image_process_result = value
        
    @property
    def slam_map(self) -> Optional[SLAMMap]:
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
                self.image_process_result = self.fetch_processed_result()
                # Sleep for the interval
                elapsed_time = time.time() - start_time
                await asyncio.sleep(max(0, self.interval - elapsed_time))
        # Run the async function in the event loop
        loop.run_until_complete(schedule_tasks())

    @abstractmethod
    async def process_image(self, image: Image.Image):
        pass
    
    @abstractmethod
    def fetch_processed_result(self) -> tuple[Image.Image, list]:
        pass

class RobotWrapper(ABC):
    def __init__(self, robot_info: RobotInfo, observation: RobotObservation, controller_func: list[callable]):
        self.robot_info = robot_info
        self.memory = RobotMemory(robot_info)
        self._observation = observation
        self._user_log = controller_func[0]
        self._probe = controller_func[1]
        common_movement_skill_func = [
            self.move,
            self.rotate,
        ]

        common_vision_skill_func = [
            self.is_visible,
            self.object_x,
            self.object_y,
            self.object_width,
            self.object_height,
            self.take_picture
        ]

        other_skills = [
            self.log,
            self.delay,
            self.probe
        ]

        self.ll_skillset: SkillSet = SkillSet.get_common_skillset(common_movement_skill_func, common_vision_skill_func, other_skills)
        self.hl_skillset: Optional[SkillSet] = None

    @abstractmethod
    def start(self) -> bool:
        pass

    @abstractmethod
    def stop(self):
        pass

    @property
    def observation(self) -> RobotObservation:
        return self._observation

    # movement skills
    @abstractmethod
    def move(self, dx: float, dy: float) -> bool:
        pass

    @abstractmethod
    def rotate(self, deg: float) -> bool:
        pass

    @abstractmethod
    def get_state(self) -> str:
        pass

    # vision skills
    def get_obj_list(self) -> list[ObjectInfo]:
        """Returns a formatted string of detected objects."""
        return self._observation.image_process_result[1] if self._observation.image_process_result else []
    
    def get_obj_list_str(self) -> str:
        """Returns a formatted string of detected objects."""
        object_list = self.get_obj_list()
        return "\n".join([str(obj) for obj in object_list]).replace("'", "")

    def get_obj_info(self, object_name: str) -> ObjectInfo:
        object_name = object_name.strip('\'').lower()

        # try to get the object info for 10 times
        for _ in range(5):
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
    
    def take_picture(self) -> bool:
        return self._user_log(self.observation.image)
    
    def log(self, message: str) -> bool:
        return self._user_log(message)

    def delay(self, sec: float) -> bool:
        time.sleep(sec)
        return True
    
    def probe(self, query: str) -> SKILL_RET_TYPE:
        return evaluate_value(self._probe(query))