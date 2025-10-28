import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict
from math import isnan

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

# ==== Utility math ====
def _make_camera_transform(forward_axis='x', up_axis='z', offset=(0, 0, 0)) -> np.ndarray:
    axis_map = {
        'x': np.array([1, 0, 0]), '-x': np.array([-1, 0, 0]),
        'y': np.array([0, 1, 0]), '-y': np.array([0, -1, 0]),
        'z': np.array([0, 0, 1]), '-z': np.array([0, 0, -1])
    }
    cam_z = axis_map[forward_axis.lower()]
    cam_y = axis_map[up_axis.lower()]
    cam_x = np.cross(cam_y, cam_z)
    cam_x = cam_x / np.linalg.norm(cam_x)
    cam_y = cam_y / np.linalg.norm(cam_y)
    cam_z = cam_z / np.linalg.norm(cam_z)
    T = np.eye(4, dtype=float)
    T[:3, 0] = cam_x
    T[:3, 1] = cam_y
    T[:3, 2] = cam_z
    T[:3, 3] = offset
    return T

def _normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(-1)
    if q.shape[0] != 4:
        raise ValueError("Quaternion must be length 4 [x,y,z,w].")
    n = np.linalg.norm(q)
    return q / (n if n > 0 else 1.0)

def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = _normalize_quat(q)
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),         2*(xz + wy)],
        [2*(xy + wz),         1 - 2*(xx + zz),     2*(yz - wx)],
        [2*(xz - wy),         2*(yz + wx),         1 - 2*(xx + yy)]
    ], dtype=float)
    return R

def _make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3,:3] = R
    T[:3, 3] = t.reshape(3)
    return T

# ==== Data containers ====
@dataclass
class TrackedObject:
    id: int
    name: str                      # base class name, e.g., "chair"
    class_idx: int                 # per-class unique number (1,2,3,…) assigned on creation
    display_name: str              # e.g., "chair#1"
    position_world: np.ndarray     # shape (3,)
    last_seen: float
    first_seen: float
    times_seen: int = 1
    velocity_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    position_camera: Optional[np.ndarray] = None  # shape (3,)
    depth_m: Optional[float] = None
    bbox: Optional[Tuple[float, float, float, float]] = None  # (cx, cy, w, h)
    # Tracking state
    status: str = "active"  # "active", "dormant", or "removed"
    missed_frames: int = 0
    confidence: float = 1.0  # confidence score (0-1)
    # Position uncertainty (1-sigma isotropic radius in meters)
    uncertainty_m: float = 0.15
    # Extra place for app-specific metadata
    attrs: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["position_world"] = self.position_world.tolist()
        d["velocity_world"] = self.velocity_world.tolist()
        if self.position_camera is not None:
            d["position_camera"] = self.position_camera.tolist()
        return d

# ==== SceneGraph ====

D435I_K = np.array([[606.1963500976562,   0.0, 320.0],
               [  0.0, 606.2576904296875, 240.0],
               [  0.0,   0.0,   1.0]], dtype=np.float32)

class SceneGraph:
    """
    Maintains a live object database from YOLO boxes + depth, producing absolute
    3D world locations and last-seen timestamps. Call .update(capture_info) per frame.
    """
    def __init__(
        self,
        rate: int = 2,
        K: np.ndarray = D435I_K,
        T_cam_in_robot: Optional[np.ndarray] = _make_camera_transform(
            forward_axis='x', up_axis='-z', offset=(0.0, 0, 0.0)
        ),
        depth_scale_m: float = 1.0,
        assoc_dist_m: float = 0.5,
        alpha_pos: float = 0.6,
        stale_after_s: float = 50.0,
        max_missed_frames: int = 10,
        dormant_timeout_s: float = 300.0,
        revisit_dist_m: float = 0.8,
        # NEW: clustering + uncertainty
        merge_dist_m: float = 1.0,             # detections of same class within this radius → one
        uncertainty_alpha: float = 0.3,        # smoothing for uncertainty updates
        min_uncertainty_m: float = 0.05,       # floor (1-sigma)
        base_meas_noise_m: float = 0.05        # baseline sensor noise added each update
    ):
        self.rate = rate
        self._last_update_time = 0.0
        self.K = np.asarray(K, dtype=float)
        if self.K.shape != (3, 3):
            raise ValueError("K must be 3x3 pinhole intrinsics.")
        self.fx, self.fy = self.K[0, 0], self.K[1, 1]
        self.cx, self.cy = self.K[0, 2], self.K[1, 2]

        self.T_cam_in_robot = np.eye(4) if T_cam_in_robot is None else np.asarray(T_cam_in_robot, dtype=float)
        self.depth_scale_m = float(depth_scale_m)
        self.assoc_dist_m = float(assoc_dist_m)
        self.alpha_pos = float(alpha_pos)
        self.stale_after_s = float(stale_after_s)
        self.max_missed_frames = int(max_missed_frames)
        self.dormant_timeout_s = float(dormant_timeout_s)
        self.revisit_dist_m = float(revisit_dist_m)

        self.merge_dist_m = float(merge_dist_m)
        self.uncertainty_alpha = float(uncertainty_alpha)
        self.min_uncertainty_m = float(min_uncertainty_m)
        self.base_meas_noise_m = float(base_meas_noise_m)

        self.objects: Dict[int, TrackedObject] = {}
        self.ids_by_name: Dict[str, List[int]] = defaultdict(list)  # keyed by *base* class name
        self._next_id = 1
        self._frame_count = 0
        self._class_counters: Dict[str, int] = defaultdict(int)     # per-class
