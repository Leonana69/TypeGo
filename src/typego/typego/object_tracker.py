from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import time, math, itertools
import numpy as np
from scipy.spatial.transform import Rotation as R
try:
    from scipy.optimize import linear_sum_assignment
    _HAS_LSA = True
except Exception:
    _HAS_LSA = False

# ----------------------------------------------------
# Camera intrinsics (fill with your values on startup)
# ----------------------------------------------------
@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    depth_unit_m: float = 0.001  # your depth is mm → meters

# ----------------
# Math helpers
# ----------------
def _make_T(t_xyz: np.ndarray, q_xyzw: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_quat(q_xyzw).as_matrix()
    T[:3, 3] = t_xyz
    return T

def _pixel_to_cam(u: float, v: float, Z_m: float, K: CameraIntrinsics) -> np.ndarray:
    X = (u - K.cx) / K.fx * Z_m
    Y = (v - K.cy) / K.fy * Z_m
    return np.array([X, Y, Z_m], dtype=np.float64)

def _clamp(v, lo, hi): return max(lo, min(hi, v))

def _box_norm_to_px(cx: float, cy: float, w: float, h: float, W: int, H: int) -> Tuple[int,int,int,int]:
    x1 = int(_clamp((cx - w/2.0) * W, 0, W-1))
    y1 = int(_clamp((cy - h/2.0) * H, 0, H-1))
    x2 = int(_clamp((cx + w/2.0) * W, 0, W-1))
    y2 = int(_clamp((cy + h/2.0) * H, 0, H-1))
    return x1, y1, x2, y2

def _iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1 + 1), max(0, inter_y2 - inter_y1 + 1)
    inter = iw * ih
    a_area = (ax2 - ax1 + 1) * (ay2 - ay1 + 1)
    b_area = (bx2 - bx1 + 1) * (by2 - by1 + 1)
    union = a_area + b_area - inter + 1e-9
    return inter / union

def _robust_depth_mm(depth: np.ndarray, box_px: Tuple[int,int,int,int],
                     center_frac: float = 0.3,
                     min_mm: float = 100.0, max_mm: float = 6000.0) -> Optional[float]:
    x1,y1,x2,y2 = box_px
    w = x2 - x1 + 1; h = y2 - y1 + 1
    cx = x1 + 0.5*w;  cy = y1 + 0.5*h
    dx = max(2, int(w * center_frac * 0.5))
    dy = max(2, int(h * center_frac * 0.5))
    sx1 = max(x1, int(cx - dx)); sx2 = min(x2, int(cx + dx))
    sy1 = max(y1, int(cy - dy)); sy2 = min(y2, int(cy + dy))
    patch = depth[sy1:sy2+1, sx1:sx2+1].astype(np.float32)
    if patch.size == 0: return None
    valid = patch[(patch >= min_mm) & (patch <= max_mm)]
    if valid.size < 5: return None
    return float(np.median(valid))

# -----------------------------
# Simple constant-velocity KF
# -----------------------------
class _KF3D:
    """
    State x = [X Y Z VX VY VZ]^T in MAP frame.
    Measurement z = [X Y Z]^T (position).
    """
    def __init__(self, meas_std=0.05, accel_std=0.2):
        self.x = np.zeros((6,1), dtype=np.float64)
        self.P = np.diag([0.3,0.3,0.3, 1.0,1.0,1.0])  # initial covariance
        self.R = (meas_std ** 2) * np.eye(3)
        self.accel_std = accel_std
        self.initialized = False

    def _F_Q(self, dt: float):
        F = np.eye(6)
        F[0,3] = F[1,4] = F[2,5] = dt
        q = self.accel_std ** 2
        dt2 = dt*dt; dt3 = dt2*dt
        Qp  = (dt3/3.0) * q * np.eye(3)
        Qpv = (dt2/2.0) * q * np.eye(3)
        Qv  = dt * q * np.eye(3)
        Q = np.block([[Qp,  Qpv],
                      [Qpv, Qv ]])
        return F, Q

    def predict(self, dt: float):
        if not self.initialized: return
        F, Q = self._F_Q(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z_pos_map: np.ndarray):
        if not self.initialized:
            self.x[:3,0] = z_pos_map.reshape(3)
            self.initialized = True
            return
        z = z_pos_map.reshape(3,1)
        H = np.zeros((3,6)); H[0,0]=H[1,1]=H[2,2]=1.0
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(6)
        self.P = (I - K @ H) @ self.P

    def mean(self) -> np.ndarray:
        return self.x[:,0].copy()

# -----------------------------
# Track and detection structs
# -----------------------------
@dataclass
class _Meas:
    name: str
    bbox_px: Tuple[int,int,int,int]
    has_3d: bool
    p_map: Optional[np.ndarray]  # (3,) in meters
    center_px: Tuple[float,float]

class _Track3D:
    _next_id = itertools.count(1)
    def __init__(self, name: str, p_map_init: np.ndarray, bbox_px: Tuple[int,int,int,int]):
        self.id: int = next(_Track3D._next_id)
        self.name: str = name
        self.kf = _KF3D(meas_std=0.05, accel_std=0.15)
        self.kf.update(p_map_init)
        self.last_confirmed_map: np.ndarray = p_map_init.copy()
        self.last_bbox_px = bbox_px
        self.last_update_time = time.time()
        self.last_step_time = self.last_update_time
        self.age = 1
        self.misses = 0
        self.hits = 1

    def predict(self, now: float):
        dt = max(1e-3, now - self.last_step_time)
        self.kf.predict(dt)
        self.last_step_time = now

    def update_on_detection(self, p_map: np.ndarray, bbox_px: Tuple[int,int,int,int]):
        self.kf.update(p_map)
        self.last_confirmed_map = p_map.copy()  # <-- only change on detection
        self.last_bbox_px = bbox_px
        self.last_update_time = time.time()
        self.hits += 1
        self.misses = 0

    # Accessors
    def predicted_map(self) -> np.ndarray:
        return self.kf.mean()[:3]

# ---------------------------------
# Multi-object 3D tracker (map)
# ---------------------------------
class MultiObjectTracker3D:
    """
    - Keeps a track per instance (many per class supported).
    - Only commits location updates when a detection is assigned.
    - Association uses 3D distance (if both sides have 3D), otherwise 2D IoU / center distance.
    """
    def __init__(self,
                 cam_K: CameraIntrinsics,
                 T_robot_cam: Optional[np.ndarray] = None,
                 max_age: int = 15,              # frames without detection before deletion
                 dist_gate_m: float = 1.5,       # 3D gating
                 iou_gate: float = 0.05,         # 2D IoU gating (very loose)
                 center_px_gate: float = 120.0): # 2D center distance gate (pixels)
        self.K = cam_K
        self.T_robot_cam = np.eye(4) if T_robot_cam is None else T_robot_cam.astype(np.float64)
        self.max_age = max_age
        self.dist_gate_m = dist_gate_m
        self.iou_gate = iou_gate
        self.center_px_gate = center_px_gate
        self.tracks: Dict[int, _Track3D] = {}

    # ---------- public API ----------
    def step(self, obs) -> List[dict]:
        """
        Process one frame:
        - Predict all tracks (internally).
        - Build measurements from detections (only those with valid 3D will create/commit).
        - Associate by class, update matched tracks.
        - Age / prune unmatched tracks.
        Returns a list of current track dicts.
        """
        now = time.time()
        for tr in self.tracks.values():
            tr.predict(now)

        meas_list = self._measurements_from_obs(obs)  # includes 2D-only (has_3d=False)
        # Associate/dict by class to avoid cross-class matches
        by_cls: Dict[str, Tuple[List[int], List[int]]] = {}  # name -> (track_idx_list, meas_idx_list)
        track_ids = list(self.tracks.keys())
        for name in set([m.name for m in meas_list] + [self.tracks[i].name for i in track_ids]):
            by_cls[name] = ([], [])
        for idx, tid in enumerate(track_ids):
            by_cls[self.tracks[tid].name][0].append(tid)
        for midx, m in enumerate(meas_list):
            by_cls[m.name][1].append(midx)

        matched_tids = set()
        matched_midx = set()

        for name, (tids, mids) in by_cls.items():
            if not tids or not mids:
                continue
            # Build cost matrix
            C = np.full((len(tids), len(mids)), fill_value=1e6, dtype=np.float64)
            for i, tid in enumerate(tids):
                tr = self.tracks[tid]
                pred_map = tr.predicted_map()
                last_box = tr.last_bbox_px
                # Precompute last box center
                lcx = 0.5*(last_box[0]+last_box[2]); lcy = 0.5*(last_box[1]+last_box[3])
                for j, midx in enumerate(mids):
                    m = meas_list[midx]
                    # Default: huge cost
                    cost = 1e6
                    # 3D distance if both sides have 3D (track always has 3D state; meas may not)
                    if m.has_3d and m.p_map is not None:
                        d = np.linalg.norm(pred_map - m.p_map)
                        if d <= self.dist_gate_m:
                            cost = d  # meters
                    else:
                        # Fallback 2D association with IoU/center distance gates
                        iou = _iou(last_box, m.bbox_px)
                        cdist = math.hypot(lcx - m.center_px[0], lcy - m.center_px[1])
                        if iou >= self.iou_gate or cdist <= self.center_px_gate:
                            # Translate to a cost; prefer high IoU, small center distance
                            cost = (1.0 - min(iou, 1.0)) * 10.0 + cdist * 0.01
                    C[i, j] = cost

            # Solve assignment
            if _HAS_LSA:
                ri, cj = linear_sum_assignment(C)
                pairs = [(tids[i], mids[j]) for i, j in zip(ri, cj)]
            else:
                pairs = []
                Cm = C.copy()
                used_i, used_j = set(), set()
                while True:
                    i, j = divmod(np.argmin(Cm), Cm.shape[1])
                    if i in used_i or j in used_j: break
                    if Cm[i, j] >= 1e5: break
                    used_i.add(i); used_j.add(j)
                    pairs.append((tids[i], mids[j]))
                    Cm[i, :] = 1e6; Cm[:, j] = 1e6

            # Apply gating and update
            for tid, midx in pairs:
                m = meas_list[midx]
                # Only count as a match if cost under gates (already enforced in C)
                # Only UPDATE track (commit new map) if measurement has 3D
                if m.has_3d and m.p_map is not None:
                    self.tracks[tid].update_on_detection(m.p_map, m.bbox_px)
                    matched_tids.add(tid)
                    matched_midx.add(midx)

            # For 2D-only matches we still "reserve" the detection, but don't update map
            for tid, midx in pairs:
                if midx in matched_midx: 
                    continue
                m = meas_list[midx]
                if not m.has_3d:
                    matched_tids.add(tid)
                    matched_midx.add(midx)

        # Unmatched tracks age
        for tid, tr in self.tracks.items():
            if tid not in matched_tids:
                tr.misses += 1
                tr.age += 1

        # Create new tracks for unmatched 3D measurements
        for midx, m in enumerate(meas_list):
            if midx in matched_midx:
                continue
            if m.has_3d and m.p_map is not None:
                new_tr = _Track3D(m.name, m.p_map, m.bbox_px)
                self.tracks[new_tr.id] = new_tr

        # Prune stale
        stale = [tid for tid, tr in self.tracks.items() if tr.misses > self.max_age]
        for tid in stale:
            del self.tracks[tid]

        # Export public view (positions only reflect last confirmed detection)
        out = []
        for tid, tr in self.tracks.items():
            out.append({
                "id": tid,
                "name": tr.name,
                "position_map_m": tr.last_confirmed_map.copy(),  # unchanged unless detection
                "last_update_time": tr.last_update_time,
                "stale_age_s": time.time() - tr.last_update_time,
                "bbox_px": tr.last_bbox_px,
                "age_frames": tr.age,
                "misses": tr.misses,
                "hits": tr.hits,
            })
        return out

    # ---------- internals ----------
    def _measurements_from_obs(self, obs) -> List[_Meas]:
        """
        Convert YOLO ObjectInfo list into measurement list.
        We accept 2D-only detections (has_3d=False) to help association,
        but only 3D ones (with valid depth) can update/create tracks.
        """
        latest = getattr(obs.yolo_client, "latest_result", None)
        if latest is None:
            return []
        _, object_list = latest
        if not object_list:
            return []

        W, H = self.K.width, self.K.height
        depth = getattr(obs, "depth", None)
        # Build transforms
        T_map_odom   = _make_T(obs.map2odom_translation,  obs.map2odom_rotation)
        T_odom_robot = _make_T(obs.odom2robot_translation, obs.odom2robot_rotation)
        T_map_robot  = T_map_odom @ T_odom_robot

        out: List[_Meas] = []
        for o in object_list:
            name = getattr(o, "name", str(getattr(o, "cls", ""))).lower()
            cx, cy, w, h = float(o.cx), float(o.cy), float(o.w), float(o.h)
            x1,y1,x2,y2 = _box_norm_to_px(cx, cy, w, h, W, H)
            center_px = (0.5*(x1+x2), 0.5*(y1+y2))

            # Try to get depth (meters)
            Z_m: Optional[float] = None
            if getattr(o, "depth", None) is not None:
                Z_m = (o.depth * 0.001) if o.depth > 10.0 else float(o.depth)
            elif depth is not None and depth.shape == (H, W):
                d_mm = _robust_depth_mm(depth, (x1,y1,x2,y2))
                if d_mm is not None:
                    Z_m = d_mm * self.K.depth_unit_m

            if Z_m is not None and Z_m > 0:
                # back-project to camera
                p_cam = _pixel_to_cam(center_px[0], center_px[1], Z_m, self.K)
                p_cam_h = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0], dtype=np.float64)
                # camera → map
                p_map_h = T_map_robot @ (self.T_robot_cam @ p_cam_h)
                p_map = p_map_h[:3] / max(1e-9, p_map_h[3])
                out.append(_Meas(name=name, bbox_px=(x1,y1,x2,y2), has_3d=True, p_map=p_map, center_px=center_px))
            else:
                # 2D-only measurement (for association only)
                out.append(_Meas(name=name, bbox_px=(x1,y1,x2,y2), has_3d=False, p_map=None, center_px=center_px))

            # Note: if you *never* want 2D-only association, remove the else-branch above.

        return out
