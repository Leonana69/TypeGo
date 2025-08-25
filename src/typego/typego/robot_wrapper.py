from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Iterable
from numpy import ndarray
import time, threading
from PIL import Image
import asyncio, cv2
import re
import numpy as np
from collections import deque
from enum import Enum, auto
import inspect

from typego.robot_info import RobotInfo
from typego.yolo_client import ObjectInfo
from typego.skill_item import SkillRegistry
from typego_interface.msg import WayPointArray, WayPoint
from typego.auto_lock_properties import auto_locked_properties

class SubSystem(Enum):
    NONE = auto()
    MOVEMENT = auto()
    SOUND = auto()


class RobotPosture(Enum):
    UNINIT = auto()
    STANDING = auto()
    LYING = auto()
    MOVING = auto()
    GRABBING = auto()
    RELEASED = auto()

    @classmethod
    def from_string(cls, s: str):
        try:
            return cls[s.upper()]  # Matches "standing" -> STANDING, etc.
        except KeyError:
            raise ValueError(f"Unknown posture: {s}")


@dataclass(slots=True)
class SLAMMap:
    # Core map
    map_data: Optional[ndarray[np.int16]] = None
    width: int = 0
    height: int = 0
    origin: tuple[float, float] = (0.0, 0.0)      # (x0, y0) in world coords (meters)
    resolution: float = 0.0                       # meters/pixel
    inv_resolution: float = 0.0                   # precomputed for speed

    # Robot state
    robot_loc: tuple[float, float] = (0.0, 0.0)   # (x, y) world meters
    robot_yaw: float = 0.0                        # radians

    # Waypoints
    waypoints: Optional["WayPointArray"] = None
    _wp_ids: Optional[ndarray[np.int32]] = None
    _wp_xy: Optional[ndarray[np.float32]] = None
    _wp_index_by_id: dict[int, int] = field(default_factory=dict)

    # Trajectory: (timestamp, (x,y)), sliding 60s window
    trajectory: deque[tuple[float, tuple[float, float]]] = field(
        default_factory=lambda: deque(maxlen=4096)
    )

    # Caching / threading
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _base_bgr: Optional[ndarray[np.uint8]] = None   # cached colorized occupancy
    _last_prune_ts: float = 0.0

    # ------------- Update methods -------------
    def update_map(
        self,
        map_data: ndarray[np.int16],
        width: int,
        height: int,
        origin: tuple[float, float],
        resolution: float,
    ) -> None:
        """Replace the occupancy grid and invalidate cache."""
        with self._lock:
            self.map_data = map_data
            self.width = int(width)
            self.height = int(height)
            self.origin = (float(origin[0]), float(origin[1]))
            self.resolution = float(resolution)
            self.inv_resolution = 0.0 if resolution == 0 else 1.0 / resolution
            self._base_bgr = None  # invalidate cached visualization

    def update_waypoints(self, waypoints: "WayPointArray") -> None:
        """Update waypoints and build fast lookup structures."""
        with self._lock:
            self.waypoints = waypoints
            if waypoints is None or not getattr(waypoints, "waypoints", None):
                self._wp_ids = None
                self._wp_xy = None
                self._wp_index_by_id.clear()
                return

            wps = waypoints.waypoints
            self._wp_ids = np.fromiter((wp.id for wp in wps), count=len(wps), dtype=np.int32)
            self._wp_xy = np.array([(wp.x, wp.y) for wp in wps], dtype=np.float32)
            self._wp_index_by_id = {wp.id: i for i, wp in enumerate(wps)}

    def update_robot_state(self, robot_loc: tuple[float, float], robot_yaw: float) -> None:
        """Set robot pose and append to trajectory (auto-prune to 60s)."""
        now = time.time()
        with self._lock:
            self.robot_loc = (float(robot_loc[0]), float(robot_loc[1]))
            self.robot_yaw = float(robot_yaw)
            self.trajectory.append((now, self.robot_loc))
            # prune at most ~10 Hz to amortize cost
            if now - self._last_prune_ts > 0.1:
                cutoff = now - 60.0
                while self.trajectory and self.trajectory[0][0] < cutoff:
                    self.trajectory.popleft()
                self._last_prune_ts = now

    # ------------- Queries -------------
    def get_waypoint(self, id: int) -> Optional["WayPoint"]:
        with self._lock:
            if self.waypoints is None or not self._wp_index_by_id:
                return None
            idx = self._wp_index_by_id.get(int(id))
            if idx is None:
                return None
            return self.waypoints.waypoints[idx]

    def get_nearest_waypoint_id(self, loc: ndarray[np.float32] | Iterable[float]) -> int | None:
        """Vectorized nearest search (O(N)) without Python loops."""
        with self._lock:
            if self._wp_xy is None or self._wp_xy.size == 0:
                return None
            xy = np.asarray(loc, dtype=np.float32)
            if xy.shape[0] >= 2:
                xy = xy[:2]
            else:
                xy = np.pad(xy, (0, 2 - xy.shape[0]))
            diff = self._wp_xy - xy  # (N,2)
            # argmin of squared distance (no sqrt)
            idx = int(np.argmin(np.einsum("ij,ij->i", diff, diff)))
            return int(self._wp_ids[idx]) if self._wp_ids is not None else None

    def get_waypoint_list_str(self) -> list[dict]:
        with self._lock:
            if self.waypoints is None:
                return []
            return [{"id": wp.id, "label": wp.label} for wp in self.waypoints.waypoints]

    def is_empty(self) -> bool:
        with self._lock:
            return self.map_data is None or self.map_data.size == 0

    # ------------- Visualization -------------
    def get_map(
        self,
        *,
        scale: float = 2.0,
        draw_waypoints: bool = True,
        draw_robot: bool = True,
        draw_ids: bool = True,
        arrow_len_px: int = 12,
    ) -> Optional[ndarray[np.uint8]]:
        """
        Returns a BGR image of the occupancy map with overlays (robot, waypoints).
        Uses cached colorized base for speed; overlays are re-drawn each call.
        """
        with self._lock:
            if self.map_data is None or self.width == 0 or self.height == 0:
                return None

            base = self._rebuild_base_bgr_if_needed().copy()

            # Draw robot
            if draw_robot:
                u, v = self.world_to_pixel(self.robot_loc[0], self.robot_loc[1])
                cv2.circle(base, (u, v), 3, (0, 255, 0), -1)
                du = int(arrow_len_px * np.cos(self.robot_yaw))
                dv = int(arrow_len_px * np.sin(self.robot_yaw))
                cv2.arrowedLine(base, (u, v), (u + du, v - dv), (0, 255, 0), 1, tipLength=0.3)

            # Draw waypoints
            if draw_waypoints and self.waypoints is not None:
                for wp in self.waypoints.waypoints:
                    u, v = self.world_to_pixel(wp.x, wp.y)
                    cv2.circle(base, (u, v), 3, (255, 0, 0), -1)
                    if draw_ids:
                        cv2.putText(base, f"{wp.id}", (u + 4, v - 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)

            if scale != 1.0:
                base = cv2.resize(base, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            return base

    # ------------- Helpers -------------
    def world_to_pixel(self, x: float, y: float) -> tuple[int, int]:
        """World meters → image pixel (OpenCV coords; origin at top-left)."""
        u = int((x - self.origin[0]) * self.inv_resolution)
        v = self.height - int((y - self.origin[1]) * self.inv_resolution)
        # clip to image bounds
        if u < 0: u = 0
        elif u >= self.width: u = self.width - 1
        if v < 0: v = 0
        elif v >= self.height: v = self.height - 1
        return u, v

    def _rebuild_base_bgr_if_needed(self) -> ndarray[np.uint8]:
        """Colorize occupancy grid once and cache it (BGR, flipped for OpenCV)."""
        if self._base_bgr is not None:
            return self._base_bgr
        # Map: free(0)->255, unknown(-1)->127, occupied(>0)->0
        md = self.map_data
        # Build grayscale efficiently without chained boolean writes
        gray = np.full((self.height, self.width), 0, dtype=np.uint8)
        if md is None:
            self._base_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            return self._base_bgr
        # Note: md values typically in {-1, 0, 100}
        gray[md == 0] = 255
        gray[md == -1] = 127
        # occupied already 0
        gray = cv2.flip(gray, 0)  # align with world-to-pixel formula above
        self._base_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return self._base_bgr


@auto_locked_properties(
    copy_on_get={"orientation", "position"},                      # return defensive copies
    set_cast={
        "orientation": lambda a: np.asarray(a, dtype=np.float32), # normalize dtype on set
        "position":    lambda a: np.asarray(a, dtype=np.float32),
    }
)
class RobotObservation(ABC):
    # Declare private fields once; the decorator discovers these automatically
    _image: Optional[Image.Image]
    _depth: Optional[ndarray[np.float32]]
    _orientation: ndarray[np.float32]
    _position: ndarray[np.float32]
    _slam_map: "SLAMMap"
    _posture: RobotPosture
    def __init__(self, robot_info: RobotInfo, rate: int,
                 *,
                 consumer_concurrency: int = 1,
                 queue_size: int = 1,
                 copy_on_enqueue: bool = False):
        self.interval = 1.0 / rate
        self.robot_info = robot_info

        self._image = None
        self._depth = None
        self._orientation = np.array([0.0, 0.0, 0.0])
        self._position = np.array([0.0, 0.0, 0.0])
        self._slam_map = SLAMMap()
        self._posture = RobotPosture.UNINIT

        # Async infra
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_evt = threading.Event()

        # Tuning knobs
        self._consumer_concurrency = max(1, consumer_concurrency)
        self._queue_size = max(1, queue_size)
        self._copy_on_enqueue = copy_on_enqueue

    # -------------------------
    # Lifecycle
    # -------------------------
    def start(self):
        if self._thread is not None:
            return  # idempotent
        self._stop_evt.clear()
        self._start()
        self._thread = threading.Thread(target=self._run_loop_thread, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return  # idempotent
        self._stop_evt.set()
        # Wake the loop if it's sleeping:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(lambda: None)
        self._thread.join()
        self._thread = None
        self._stop()

    def _run_loop_thread(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._main())
        finally:
            # Cancel anything left and close loop cleanly
            pending = asyncio.all_tasks(self._loop)
            for t in pending:
                t.cancel()
            if pending:
                self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self._loop.close()
            self._loop = None

    async def _main(self):
        # Single-producer (periodic) + N consumers (bounded concurrency)
        queue: asyncio.Queue[Image.Image] = asyncio.Queue(maxsize=self._queue_size)

        producer = asyncio.create_task(self._producer(queue))
        consumers = [asyncio.create_task(self._consumer(queue)) for _ in range(self._consumer_concurrency)]

        # Wait until stop is requested from another thread
        await asyncio.to_thread(self._stop_evt.wait)

        # Stop producer first, then drain
        producer.cancel()
        await asyncio.gather(producer, return_exceptions=True)
        await queue.join()  # let consumers finish what's queued

        # Cancel consumers and wait
        for c in consumers:
            c.cancel()
        await asyncio.gather(*consumers, return_exceptions=True)

    async def _producer(self, queue: "asyncio.Queue[Image.Image]"):
        interval = self.interval
        while not self._stop_evt.is_set():
            t0 = time.perf_counter()
            img = self.image  # property getter (thread-safe)
            if img is not None:
                if self._copy_on_enqueue:
                    # If your camera reuses buffers, this avoids aliasing
                    try:
                        img = img.copy()
                    except Exception:
                        pass
                # Keep latest only: if full, drop the oldest
                if queue.full():
                    try:
                        queue.get_nowait()
                        queue.task_done()
                    except asyncio.QueueEmpty:
                        pass
                await queue.put(img)
            # Periodic pacing
            dt = time.perf_counter() - t0
            # Sleep the remainder; avoid negative
            await asyncio.sleep(interval - dt if dt < interval else 0.0)

    async def _consumer(self, queue: "asyncio.Queue[Image.Image]"):
        while True:
            img = await queue.get()
            try:
                await self.process_image(img)
            finally:
                queue.task_done()

    # -------------------------
    # Hooks for subclasses
    # -------------------------
    @abstractmethod
    def _start(self):
        ...

    @abstractmethod
    def _stop(self):
        ...

    @abstractmethod
    async def process_image(self, image: Image.Image):
        ...
    
    @abstractmethod
    def fetch_processed_result(self) -> tuple[Image.Image, list] | None:
        ...
    
    @abstractmethod
    def obs(self) -> dict:
        ...

    def blocked(self) -> bool:
        return False
    
    def fetch_command(self) -> str | None:
        """Fetch the command from other sources."""
        return None

def robot_skill(name: str, description: str = "", subsystem: SubSystem = SubSystem.NONE):
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
        self.observation = observation
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

    # movement skills
    # @abstractmethod
    # @robot_skill("move", description="Move by (dx, dy) m distance (dx: +forward, dy: +left)")
    # def move(self, dx: float, dy: float) -> bool:
    #     pass

    @abstractmethod
    @robot_skill("move_forward", description="Move forward by a certain distance (m)", subsystem=SubSystem.MOVEMENT)
    def move_forward(self, distance: float) -> bool:
        pass

    @abstractmethod
    @robot_skill("move_back", description="Move back by a certain distance (m)", subsystem=SubSystem.MOVEMENT)
    def move_back(self, distance: float) -> bool:
        pass

    @abstractmethod
    @robot_skill("move_left", description="Move left by a certain distance (m)", subsystem=SubSystem.MOVEMENT)
    def move_left(self, distance: float) -> bool:
        pass

    @abstractmethod
    @robot_skill("move_right", description="Move right by a certain distance (m)", subsystem=SubSystem.MOVEMENT)
    def move_right(self, distance: float) -> bool:
        pass

    # @abstractmethod
    # @robot_skill("rotate", description="Rotate by deg degrees (deg: +left or clockwise)")
    # def rotate(self, deg: float) -> bool:
    #     pass

    @abstractmethod
    @robot_skill("turn_left", description="Rotate counter-clockwise by a certain angle (degrees)", subsystem=SubSystem.MOVEMENT)
    def turn_left(self, deg: float) -> bool:
        pass

    @abstractmethod
    @robot_skill("turn_right", description="Rotate clockwise by a certain angle (degrees)", subsystem=SubSystem.MOVEMENT)
    def turn_right(self, deg: float) -> bool:
        pass

    # vision skills
    def get_obj_list(self) -> list[ObjectInfo]:
        """Returns a formatted string of detected objects."""
        process_result = self.observation.fetch_processed_result()
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