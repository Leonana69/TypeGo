from typing import Optional, Iterable
from abc import ABC, abstractmethod
from numpy import ndarray
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from PIL import Image
from json import JSONEncoder
import time, threading
import asyncio, cv2
import numpy as np

from typego.robot_info import RobotInfo
from typego.auto_lock_properties import auto_locked_properties
from typego_interface.msg import WayPointArray, WayPoint
from typego.yolo_client import ObjectBox

class RobotPosture(str, Enum):
    """
    Robot posture states.
    """
    UNINIT = 'uninit'
    STANDING = 'standing'
    LYING = 'lying'
    MOVING = 'moving'
    GRABBING = 'grabbing'
    RELEASED = 'released'

    @classmethod
    def from_string(cls, s: str):
        try:
            return cls[s.upper()]
        except KeyError:
            raise ValueError(f"Unknown posture: {s}")

class ObservationEncoder(JSONEncoder):
    """Custom JSON encoder for ObjectBox class"""
    def default(self, obj):
        if isinstance(obj, ObjectBox):
            return obj.to_dict()
        elif isinstance(obj, (np.float32, np.float64, float)):
            return round(float(obj), 2)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

@auto_locked_properties(
    copy_on_get={"orientation", "position"},                      # return defensive copies
    set_cast={
        "orientation": lambda a: np.asarray(a, dtype=np.float32), # normalize dtype on set
        "position":    lambda a: np.asarray(a, dtype=np.float32),
    }
)
class RobotObservation(ABC):
    """
    Base class for robot observations.
    Subclasses should implement the abstract methods to provide specific functionalities.
    """
    # Declare private fields once; the decorator discovers these automatically
    _rgb_image: Optional[Image.Image]
    _depth_image: Optional[ndarray[np.float32]]
    _orientation: ndarray[np.float32]
    _position: ndarray[np.float32]
    _slam_map: "SLAMMap"
    _posture: RobotPosture
    _command: Optional[str]
    def __init__(self, robot_info: RobotInfo, rate: int,
                 *,
                 consumer_concurrency: int = 1,
                 queue_size: int = 1,
                 copy_on_enqueue: bool = False):
        self.interval = 1.0 / rate
        self.robot_info = robot_info

        self.running = False

        self._rgb_image = None
        self._depth_image = None
        self._orientation = np.array([0.0, 0.0, 0.0])
        self._position = np.array([0.0, 0.0, 0.0])
        self._slam_map = SLAMMap()
        self._posture = RobotPosture.UNINIT
        self._command = None

        # Async infra
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_evt = threading.Event()

        # Tuning knobs
        self._consumer_concurrency = max(1, consumer_concurrency)
        self._queue_size = max(1, queue_size)
        self._copy_on_enqueue = copy_on_enqueue

    # ---- Lifecycle ----
    def start(self):
        if self.running:
            raise RuntimeError("Observation is already running")
        self.running = True
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run_loop_thread, daemon=True)
        self._thread.start()
        self._start()

    def stop(self):
        if not self.running:
            return
        self.running = False
        self._stop()
        self._stop_evt.set()
        # Wake the loop if it's sleeping:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(lambda: None)
        self._thread.join()
        self._thread = None

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
        """
        Main async task: producer + N consumers.
        """
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
        """
        Periodic producer: capture image and enqueue.
        """
        interval = self.interval
        while not self._stop_evt.is_set():
            t0 = time.perf_counter()
            img = self.rgb_image  # property getter (thread-safe)
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
        """
        Consumer: fetch image from queue and process it.
        """
        while True:
            img = await queue.get()
            try:
                await self.process_image(img)
            finally:
                queue.task_done()

    # ---- Abstract methods ----
    @abstractmethod
    def _start(self):
        """Called once when starting observation."""
        ...

    @abstractmethod
    def _stop(self):
        """Called once when stopping observation."""
        ...

    @abstractmethod
    async def process_image(self, image: Image.Image):
        """Process a new image get from the robot capture."""
        ...
    
    @abstractmethod
    def fetch_objects(self) -> tuple[Image.Image, list[ObjectBox]] | None:
        ...
    
    @abstractmethod
    def obs(self) -> dict:
        """Return the full observation as a dictionary."""
        ...

    def obs_str(self) -> str:
        """Return the full observation as a JSON string."""
        return ObservationEncoder().encode(self.obs())

    def blocked(self) -> bool:
        return False
    
    def fetch_command(self) -> str | None:
        """Fetch the command from other sources."""
        return None
    
    def get_obj_info(self, object_name: str, reliable=False) -> ObjectBox | None:
        object_name = object_name.strip('\'').strip('"').lower()

        for _ in range(3):
            rslt = self.fetch_objects()
            object_list = []
            if rslt:
                object_list = rslt[1]

            for obj in object_list:
                if obj.name.startswith(object_name):
                    return obj
            if not reliable:
                break
            time.sleep(0.2)
        return None

@dataclass(slots=True)
class SLAMMap:
    """
    SLAM map representation and utilities.
    """
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

    # ---- Update methods ----
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

    # ---- Queries ----
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

    # ---- Visualization -----
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

    # ---- Helpers -----
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
