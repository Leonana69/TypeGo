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
    """
    Helper to create a camera transform for common mounting configurations.
    
    Args:
        forward_axis: which robot axis the camera points along ('x', 'y', 'z', '-x', '-y', '-z')
        up_axis: which robot axis is "up" for the camera ('-y' common for forward-looking)
        offset: (x, y, z) position offset of camera from robot base in meters
    
    Returns:
        4x4 transform matrix
    
    Example:
        # Camera looking forward (robot +X), with camera Y down
        T = _make_camera_transform(forward_axis='x', up_axis='-y', offset=(0.1, 0, 0.3))
    """
    axis_map = {
        'x': np.array([1, 0, 0]), '-x': np.array([-1, 0, 0]),
        'y': np.array([0, 1, 0]), '-y': np.array([0, -1, 0]),
        'z': np.array([0, 0, 1]), '-z': np.array([0, 0, -1])
    }
    
    # Camera frame: X=right, Y=down, Z=forward (OpenCV/ROS convention)
    cam_z = axis_map[forward_axis.lower()]  # forward
    cam_y = axis_map[up_axis.lower()]        # up becomes down in camera
    cam_x = np.cross(cam_y, cam_z)           # right
    
    # Normalize in case of numerical issues
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
    # q = [x, y, z, w]
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
    name: str
    position_world: np.ndarray          # shape (3,)
    position_uncertainty: float = 0.1   # position uncertainty in meters
    last_seen: float = 0.0
    first_seen: float = 0.0
    times_seen: int = 1
    velocity_world: np.ndarray = field(default_factory=lambda: np.zeros(3))
    position_camera: Optional[np.ndarray] = None  # shape (3,)
    depth_m: Optional[float] = None
    bbox: Optional[Tuple[float, float, float, float]] = None  # (cx, cy, w, h)
    # Tracking state
    status: str = "active"  # "active", "dormant", or "removed"
    missed_frames: int = 0
    confidence: float = 1.0  # confidence score (0-1)
    # Merged object tracking
    merged_count: int = 1  # number of objects merged into this one
    # Extra place for app-specific metadata
    attrs: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = asdict(self)
        # Convert numpy arrays to python lists for JSON friendliness
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
    3D world locations and last-seen timestamps. Designed to be called each frame via .update(capture_info).

    Key parameters:
      - K: 3x3 camera intrinsics (fx, fy, cx, cy) for the color/depth pair
      - T_cam_in_robot: 4x4 transform from ROBOT base frame to CAMERA frame (robot->camera)
      - depth_scale_m: multiply raw depth units by this to get meters (e.g., 0.001 for mm)
      - assoc_dist_m: maximum 3D distance to associate to an existing object with same class
      - merge_dist_m: maximum 3D distance to merge objects of the same class
      - alpha_pos: EMA smoothing factor for position updates (0..1)
      - uncertainty_decay: decay factor for reducing uncertainty when measurements are consistent
      - uncertainty_gain: gain factor for increasing uncertainty when measurements differ
      - stale_after_s: auto-remove objects not seen for this many seconds
      - max_missed_frames: number of frames before marking object as dormant
      - dormant_timeout_s: seconds before removing dormant objects
      - revisit_dist_m: distance threshold for reactivating dormant objects
    """
    def __init__(
        self,
        rate: int = 2,
        K: np.ndarray = D435I_K,
        T_cam_in_robot: Optional[np.ndarray] = _make_camera_transform(
            forward_axis='x',     # camera Z-axis aligns with robot's +X
            up_axis='-z',         # camera Y-axis (down) aligns with robot's -Z
            offset=(0.0, 0, 0.0)  # adjust if camera is offset from robot origin
        ),
        depth_scale_m: float = 1.0,
        assoc_dist_m: float = 0.5,
        merge_dist_m: float = 1.0,
        alpha_pos: float = 0.6,
        uncertainty_decay: float = 0.02,
        uncertainty_gain: float = 0.3,
        stale_after_s: float = 600.0,
        max_missed_frames: int = 10,
        dormant_timeout_s: float = 300.0,
        revisit_dist_m: float = 0.8,
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
        self.merge_dist_m = float(merge_dist_m)
        self.alpha_pos = float(alpha_pos)
        self.uncertainty_decay = float(uncertainty_decay)
        self.uncertainty_gain = float(uncertainty_gain)
        self.stale_after_s = float(stale_after_s)
        self.max_missed_frames = int(max_missed_frames)
        self.dormant_timeout_s = float(dormant_timeout_s)
        self.revisit_dist_m = float(revisit_dist_m)

        self.objects: Dict[int, TrackedObject] = {}
        self.ids_by_name: Dict[str, List[int]] = defaultdict(list)
        self._next_id = 1
        self._frame_count = 0

        self.scene_graph_node = Node("scene_graph_node")
        self.object_publisher = self.scene_graph_node.create_publisher(MarkerArray, "tracked_objects", 10)
        self.object_marker_array = MarkerArray()

    def publish_tracked_objects(self):
        marker_array = MarkerArray()
        for obj in self.all_objects(include_dormant=True):
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.scene_graph_node.get_clock().now().to_msg()
            marker.ns = "tracked_objects"
            marker.id = obj.id
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD
            # Enhanced text with ID, name, and uncertainty
            marker.text = f"#{obj.id}: {obj.name}\n±{obj.position_uncertainty:.2f}m"
            if obj.merged_count > 1:
                marker.text += f" (×{obj.merged_count})"
            
            marker.pose.position.x = float(obj.position_world[0])
            marker.pose.position.y = float(obj.position_world[1])
            marker.pose.position.z = float(obj.position_world[2])
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            if obj.status == "active":
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0
            elif obj.status == "dormant":
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 0.5
            else:  # removed
                continue  # don't publish removed objects
            marker.lifetime.sec = 1
            marker_array.markers.append(marker)
        self.object_publisher.publish(marker_array)

    # ---------- Core API ----------
    def update(self, capture_info: Dict) -> None:
        """
        Process one capture:
          capture_info = {
            'timestamp': float,
            'position': np.array([x,y,z]),           # robot world position
            'rotation': np.array([x,y,z,w]),      # robot world rotation (quaternion)
            'image': np.ndarray (unused here),
            'depth': np.ndarray (H,W) 16u/32f,
            'yolo_result': list[ObjectBox]
          }
        """
        now = time.time()
        if now - self._last_update_time < (1.0 / self.rate):
            return
        self._last_update_time = now
        self._frame_count += 1

        ts: float = float(capture_info['timestamp'])
        robot_p = np.asarray(capture_info['position'], dtype=float).reshape(3)
        robot_q = np.asarray(capture_info['rotation'], dtype=float).reshape(4)

        # robot->world
        R_wr = _quat_to_rotmat(robot_q)
        T_wr = _make_T(R_wr, robot_p)

        # camera pose in world
        T_wc = T_wr @ self.T_cam_in_robot

        # Track which objects were seen this frame
        seen_this_frame = set()

        img_w, img_h = self.cx * 2, self.cy * 2

        # Associate & upsert
        for box in capture_info['yolo_result']:
            # accept either ObjectBox or dict-like
            name = box.name
            cx = box.cx * img_w
            cy = (box.cy * 2 if box.cy < 0.5 else (box.cy - 0.5) * 2) * img_h  # handle stacked results
            w  = box.w * img_w
            h  = box.h * img_h
            z_raw = box.dist
            if z_raw is None or isnan(z_raw):
                # No reliable depth → skip this detection
                continue

            # Convert to meters if needed
            z_m = float(z_raw * self.depth_scale_m)

            # Back-project to camera frame
            Xc = (cx - self.cx) * z_m / self.fx
            Yc = (cy - self.cy) * z_m / self.fy
            Zc = z_m
            Pc = np.array([Xc, Yc, Zc, 1.0], dtype=float)

            # Transform to world
            Pw = (T_wc @ Pc)[:3]

            # Try to associate with active objects first, then dormant
            obj_id = self._associate_multi_stage(name, Pw)
            
            if obj_id is None:
                # New object
                obj_id = self._next_id
                self._next_id += 1
                tobj = TrackedObject(
                    id=obj_id,
                    name=name,
                    position_world=Pw.copy(),
                    position_uncertainty=0.1,  # initial uncertainty
                    last_seen=ts,
                    first_seen=ts,
                    times_seen=1,
                    position_camera=np.array([Xc, Yc, Zc], dtype=float),
                    depth_m=z_m,
                    bbox=(float(cx), float(cy), float(w), float(h)),
                    status="active",
                    missed_frames=0,
                    confidence=1.0,
                    merged_count=1,
                    attrs={}
                )
                self.objects[obj_id] = tobj
                self.ids_by_name[name].append(obj_id)
            else:
                # Update existing with EMA smoothing & velocity
                o = self.objects[obj_id]
                dt = max(1e-6, ts - o.last_seen)
                
                # Calculate position difference before updating
                pos_diff = np.linalg.norm(Pw - o.position_world)
                
                # Update position with EMA
                new_pos = self.alpha_pos * Pw + (1.0 - self.alpha_pos) * o.position_world
                o.velocity_world = (new_pos - o.position_world) / dt
                o.position_world = new_pos
                
                # Update uncertainty based on measurement consistency
                # Large differences increase uncertainty, small differences decrease it
                if pos_diff > 0.1:  # threshold for "significant" difference
                    # Increase uncertainty when measurements vary
                    o.position_uncertainty = min(2.0, o.position_uncertainty + self.uncertainty_gain * pos_diff)
                else:
                    # Decrease uncertainty when measurements are consistent
                    o.position_uncertainty = max(0.05, o.position_uncertainty - self.uncertainty_decay)
                
                o.last_seen = ts
                o.times_seen += 1
                o.position_camera = np.array([Xc, Yc, Zc], dtype=float)
                o.depth_m = z_m
                o.bbox = (float(cx), float(cy), float(w), float(h))
                o.missed_frames = 0
                o.status = "active"
                # Increase confidence on repeated detections
                o.confidence = min(1.0, o.confidence + 0.1)

            seen_this_frame.add(obj_id)

        # Merge nearby objects of the same class
        self._merge_nearby_objects(ts)

        # Update objects that weren't seen this frame
        self._update_missed_objects(ts, seen_this_frame, T_wc)

        # Prune completely stale objects
        self._prune_stale(ts)

        # Publish tracked objects as ROS messages
        self.publish_tracked_objects()

    # ---------- Queries ----------

    def all_objects(self, include_dormant: bool = False) -> List[TrackedObject]:
        """Get all objects. By default, only returns active objects."""
        if include_dormant:
            return list(self.objects.values())
        return [o for o in self.objects.values() if o.status == "active"]

    def get_by_id(self, obj_id: int) -> Optional[TrackedObject]:
        return self.objects.get(obj_id)

    def get_by_class(self, name: str, include_dormant: bool = False) -> List[TrackedObject]:
        objs = [self.objects[i] for i in self.ids_by_name.get(name, []) if i in self.objects]
        if not include_dormant:
            objs = [o for o in objs if o.status == "active"]
        return objs

    def get_recent(self, within_s: float, now: Optional[float] = None) -> List[TrackedObject]:
        now = time.time() if now is None else now
        return [o for o in self.objects.values() if (now - o.last_seen) <= within_s]

    def to_jsonable(self) -> Dict:
        return {k: v.to_dict() for k, v in self.objects.items()}

    # ---------- Internals ----------

    def _associate_multi_stage(self, name: str, Pw: np.ndarray) -> Optional[int]:
        """
        Multi-stage association:
        1. Try to match with active objects (tight threshold)
        2. Try to match with dormant objects (looser threshold for revisit)
        """
        # Stage 1: Active objects with standard threshold
        candidates = [oid for oid in self.ids_by_name.get(name, [])
                     if oid in self.objects and self.objects[oid].status == "active"]
        
        best_id, best_d = None, float('inf')
        for oid in candidates:
            d = np.linalg.norm(self.objects[oid].position_world - Pw)
            if d < best_d:
                best_d, best_id = d, oid
        
        if best_d <= self.assoc_dist_m:
            return best_id

        # Stage 2: Dormant objects with revisit threshold
        dormant_candidates = [oid for oid in self.ids_by_name.get(name, [])
                             if oid in self.objects and self.objects[oid].status == "dormant"]
        
        best_id, best_d = None, float('inf')
        for oid in dormant_candidates:
            d = np.linalg.norm(self.objects[oid].position_world - Pw)
            if d < best_d:
                best_d, best_id = d, oid
        
        if best_d <= self.revisit_dist_m:
            # Reactivate dormant object
            return best_id
        
        return None

    def _merge_nearby_objects(self, ts: float) -> None:
        """
        Merge objects of the same class that are within merge_dist_m of each other.
        This handles the case where multiple instances of the same object are detected nearby.
        """
        # Group objects by class name
        for name in list(self.ids_by_name.keys()):
            active_ids = [oid for oid in self.ids_by_name[name] 
                         if oid in self.objects and self.objects[oid].status == "active"]
            
            if len(active_ids) < 2:
                continue
            
            # Find pairs to merge (use greedy approach)
            merged_this_iteration = True
            while merged_this_iteration:
                merged_this_iteration = False
                
                for i in range(len(active_ids)):
                    if active_ids[i] not in self.objects:
                        continue
                    obj1 = self.objects[active_ids[i]]
                    
                    for j in range(i + 1, len(active_ids)):
                        if active_ids[j] not in self.objects:
                            continue
                        obj2 = self.objects[active_ids[j]]
                        
                        dist = np.linalg.norm(obj1.position_world - obj2.position_world)
                        
                        if dist <= self.merge_dist_m:
                            # Merge obj2 into obj1 (keep the one seen first/more times)
                            # Prefer the object with lower ID (older object)
                            if obj1.id > obj2.id:
                                obj1, obj2 = obj2, obj1
                                active_ids[i], active_ids[j] = active_ids[j], active_ids[i]
                            
                            # Calculate weighted average position
                            total_weight = obj1.times_seen + obj2.times_seen
                            new_pos = (obj1.position_world * obj1.times_seen + 
                                     obj2.position_world * obj2.times_seen) / total_weight
                            
                            # Increase uncertainty due to position variance
                            pos_variance = np.linalg.norm(obj1.position_world - obj2.position_world)
                            obj1.position_uncertainty = max(obj1.position_uncertainty, 
                                                           obj2.position_uncertainty,
                                                           pos_variance * 0.5)
                            
                            # Update obj1 with merged information
                            obj1.position_world = new_pos
                            obj1.times_seen += obj2.times_seen
                            obj1.merged_count += obj2.merged_count
                            obj1.confidence = max(obj1.confidence, obj2.confidence)
                            obj1.last_seen = max(obj1.last_seen, obj2.last_seen)
                            
                            # Remove obj2
                            del self.objects[active_ids[j]]
                            self.ids_by_name[name] = [oid for oid in self.ids_by_name[name] 
                                                     if oid != active_ids[j]]
                            
                            # Update active_ids list
                            active_ids = [oid for oid in active_ids if oid in self.objects]
                            merged_this_iteration = True
                            break
                    
                    if merged_this_iteration:
                        break

    def _is_in_camera_view(self, obj_pos_world: np.ndarray, T_wc: np.ndarray) -> bool:
        """
        Check if an object position is within the current camera's field of view.
        
        Args:
            obj_pos_world: Object position in world coordinates (3,)
            T_wc: Camera pose in world (4x4 transform)
        
        Returns:
            True if object is in view, False otherwise
        """
        # Transform object to camera frame
        Pw = np.append(obj_pos_world, 1.0)
        T_cw = np.linalg.inv(T_wc)
        Pc = T_cw @ Pw
        
        # Check if object is in front of camera (positive Z)
        if Pc[2] <= 0.1:  # At least 10cm in front
            return False
        
        # Project to image plane
        px = (Pc[0] / Pc[2]) * self.fx + self.cx
        py = (Pc[1] / Pc[2]) * self.fy + self.cy
        
        # Check if projection is within image bounds (with small margin)
        img_w, img_h = self.cx * 2, self.cy * 2
        margin = 50  # pixels
        if px < -margin or px > img_w + margin:
            return False
        if py < -margin or py > img_h + margin:
            return False
        
        return True

    def _update_missed_objects(self, now: float, seen_ids: set, T_wc: np.ndarray) -> None:
        """Update tracking state for objects not detected this frame."""
        for oid, obj in self.objects.items():
            if oid in seen_ids or obj.status == "removed":
                continue
            
            if obj.status == "active":
                # Check if object is in current camera view
                in_view = self._is_in_camera_view(obj.position_world, T_wc)
                
                obj.missed_frames += 1
                # Decay confidence
                obj.confidence = max(0.1, obj.confidence - 0.05)
                # Increase uncertainty when object is not being observed
                obj.position_uncertainty = min(2.0, obj.position_uncertainty + 0.05)
                
                # If object is in view but not detected for multiple consecutive frames,
                # it likely disappeared. Use a threshold to handle YOLO blinking/false negatives
                if in_view and obj.missed_frames >= self.max_missed_frames:
                    obj.status = "removed"
                # Transition to dormant if missed too many frames (object out of view)
                elif not in_view and obj.missed_frames >= self.max_missed_frames:
                    obj.status = "dormant"
                    obj.missed_frames = 0

    def _prune_stale(self, now: float) -> None:
        """Remove objects that have been dormant too long."""
        to_remove = []
        
        for oid, o in self.objects.items():
            # Remove dormant objects after timeout
            if o.status == "dormant" and (now - o.last_seen) > self.dormant_timeout_s:
                to_remove.append(oid)
            # Also remove active objects that are truly stale (fallback)
            elif o.status == "active" and (now - o.last_seen) > self.stale_after_s:
                to_remove.append(oid)
        
        if not to_remove:
            return
        
        # Clean indices
        for oid in to_remove:
            name = self.objects[oid].name
            del self.objects[oid]
            self.ids_by_name[name] = [i for i in self.ids_by_name[name] if i != oid]
            if not self.ids_by_name[name]:
                del self.ids_by_name[name]