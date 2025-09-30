import time, os, math
from typing import Optional
import numpy as np
from openai import OpenAI
import threading, requests
from overrides import overrides
from PIL import Image
import cv2
import json
import queue
from enum import Enum
from scipy.spatial.transform import Rotation as R
from functools import wraps

DEBUG_MODE = False

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import Image as ROSImage
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage
from std_msgs.msg import String
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from typego.robot_wrapper import RobotWrapper, RobotObservation, RobotPosture, robot_skill
from typego.robot_info import RobotInfo
from typego.yolo_client import YoloClient
from typego.utils import print_t, ImageRecover
from typego.pid import PID
from typego_interface.msg import WayPointArray
from typego.skill_item import SubSystem

GO2_CAM_K = np.array([[818.18507419, 0.0, 637.94628188],
                      [0.0, 815.32431463, 338.3480119],
                      [0.0, 0.0, 1.0]], dtype=np.float32)

GO2_CAM_D = np.array([[-0.07203219],
                      [-0.05228525],
                      [ 0.05415833],
                      [-0.02288355]], dtype=np.float32)

class Go2Observation(RobotObservation):
    def __init__(self, robot_info: RobotInfo, node: Node, rate: int = 10):
        super().__init__(robot_info, rate)
        self.yolo_client = YoloClient(robot_info)
        self.image_recover = ImageRecover(GO2_CAM_K, GO2_CAM_D)
        self.posture = RobotPosture.STANDING
        self.init_ros_obs(node)

    def init_ros_obs(self, node: Node):
        self.map2odom_translation = np.array([0.0, 0.0, 0.0])
        self.map2odom_rotation = np.array([0.0, 0.0, 0.0, 1.0])
        self.odom2robot_translation = np.array([0.0, 0.0, 0.0])
        self.odom2robot_rotation = np.array([0.0, 0.0, 0.0, 1.0])
        # Subscribe to /camera/image_raw
        node.create_subscription(
            ROSImage,
            '/camera/image_raw',
            self._image_callback,
            QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            )
        )

        # Subscribe to /tf
        node.create_subscription(
            TFMessage,
            '/tf',
            self._tf_callback,
            QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=100,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            )
        )

        # Subscribe to /map
        node.create_subscription(
            OccupancyGrid,
            '/map',
            self._map_callback,
            10
        )

        node.create_subscription(
            WayPointArray,
            '/waypoints',
            self._waypoint_callback,
            10
        )

        # Subscribe to /scan
        node.create_subscription(
            LaserScan,
            '/scan',
            self._scan_callback,
            QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
            )
        )

        # Subscribe to /voice_command
        node.create_subscription(
            String,
            '/voice_command',
            self._voice_command_callback,
            10
        )

        self.closest_object: tuple[float, float] = (float('inf'), 0.0)  # (distance, angle)
        self.latest_command: str | None = None

    def _voice_command_callback(self, msg: String):
        """
        Handle voice commands received from the /voice_command topic.
        This is a placeholder for future implementation.
        """
        print_t(f"[Go2] Received voice command: {msg.data}")
        # You can implement command parsing and execution here
        self.command = msg.data.strip().lower()

    @overrides
    def fetch_command(self) -> str | None:
        """
        Fetch the latest command received from the /voice_command topic.
        Returns the command string or None if no command is available.
        """
        command = self.command
        self.command = None
        return command

    def _image_callback(self, msg: ROSImage):
        # Convert ROS Image message to OpenCV image
        cv_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        # Undistort the image
        frame = self.image_recover.process(cv_image)
        self.image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def _tf_callback(self, msg: TFMessage):
        def make_transform(translation, quaternion):
            T = np.eye(4)
            T[:3, :3] = R.from_quat(quaternion).as_matrix()
            T[:3, 3] = translation
            return T

        # Extract position and orientation from the TF message
        for transform in msg.transforms:
            if transform.child_frame_id == "base_link":
                self.odom2robot_translation = np.array([
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z
                ])
                self.odom2robot_rotation = np.array([
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                    transform.transform.rotation.w
                ])
            elif transform.child_frame_id == "odom":
                self.map2odom_translation = np.array([
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z
                ])
                self.map2odom_rotation = np.array([
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                    transform.transform.rotation.w
                ])

        T_map_odom = make_transform(self.map2odom_translation, self.map2odom_rotation)
        T_odom_robot = make_transform(self.odom2robot_translation, self.odom2robot_rotation)
        T_map_robot = T_map_odom @ T_odom_robot
        self.position = T_map_robot[:3, 3]
        self.orientation = R.from_matrix(T_map_robot[:3, :3]).as_euler('xyz')  # or .as_quat()
        self.slam_map.update_robot_state((self.position[0], self.position[1]), self.orientation[2])
        # print_t(f"[Go2] Position: {self.position}, Orientation: {self.orientation}")

    def _map_callback(self, msg: OccupancyGrid):
        width = msg.info.width
        height = msg.info.height
        data = np.array(msg.data, dtype=np.int8).reshape((height, width))
        self.slam_map.update_map(data, width, height, (msg.info.origin.position.x, msg.info.origin.position.y), msg.info.resolution)
        # print_t(f"[Go2] Map size: {width}x{height}, Resolution: {msg.info.resolution}")

    def _waypoint_callback(self, msg: WayPointArray):
        self.slam_map.update_waypoints(msg)

    def _scan_callback(self, msg: LaserScan):
        # Parameters with small epsilon to avoid numerical issues
        max_distance = 1.0
        min_object_width = 0.15  # 15cm
        min_object_angular_width = 0.35  # 20 degrees
        min_valid_distance = 0.001  # 1mm - ignore values below this
        angle_increment = msg.angle_increment
        
        min_distance = float('inf')
        best_object = None
        
        current_cluster = []
        
        # Helper function to process a cluster
        def process_cluster(cluster):
            nonlocal min_distance, best_object
            if not cluster:
                return
                
            # Extract just the distances from the cluster (index, distance) tuples
            cluster_distances = [d for (_, d) in cluster if d >= min_valid_distance]
            if not cluster_distances:
                return
                
            cluster_min_distance = min(cluster_distances)
            angular_width = len(cluster) * angle_increment
            physical_width = angular_width * cluster_min_distance

            if (physical_width >= min_object_width or angular_width >= min_object_angular_width) and cluster_min_distance < min_distance:
                min_distance = cluster_min_distance
                midpoint_idx = cluster[0][0] + len(cluster)//2
                best_object = {
                    'distance': cluster_min_distance,
                    'angle': msg.angle_min + midpoint_idx * angle_increment,
                    'width': physical_width,
                    'indices': (cluster[0][0], cluster[-1][0])
                }
        
        # Main processing loop
        for i, distance in enumerate(msg.ranges):
            if min_valid_distance <= distance <= max_distance:
                current_cluster.append((i, distance))
            else:
                process_cluster(current_cluster)
                current_cluster = []
        
        # Process any remaining cluster at the end
        process_cluster(current_cluster)
        
        # Output results
        if best_object:
            # print(
            #     f"Closest valid object: distance={best_object['distance']:.2f}m, "
            #     f"angle={math.degrees(best_object['angle']):.1f}°, "
            #     f"width={best_object['width']:.2f}m"
            # )
            self.closest_object = (best_object['distance'], best_object['angle'])
        else:
            # print("No valid objects found within criteria")
            self.closest_object = (float('inf'), 0.0)

    @overrides
    def blocked(self) -> bool:
        """
        Check if the robot is blocked by an obstacle.
        """
        # If the closest object is within 0.2m and it's directly in front of the robot
        distance, angle = self.closest_object
        return distance < 0.2 and abs(angle) < math.radians(60)

    @overrides
    def _start(self): return
    
    @overrides
    def _stop(self): return

    @overrides
    async def process_image(self, image: Image.Image):
        await self.yolo_client.detect(image)
    
    @overrides
    def fetch_processed_result(self) -> tuple[Image.Image, list] | None:
        return self.yolo_client.latest_result

    @overrides
    def obs(self) -> dict:
        return {
            "t": time.strftime("%H:%M:%S", time.localtime(time.time())),
            "robot": {
                "pose_world": {
                    "x": self.position[0],
                    "y": self.position[1],
                    "yaw": self.orientation[2]
                },
                "posture": self.posture.name.lower(),
            },
            "perception": self.yolo_client.latest_result[1] if self.yolo_client.latest_result else [],
            "nav": {
                "current_wp": self.slam_map.get_nearest_waypoint_id(self.position),
                "waypoints": self.slam_map.get_waypoint_list_str()
            }
        }

class Go2Action:
    def __init__(self, actions: list[tuple[callable, float]]):
        self.actions = actions  # List of tuples (function, duration)
        
    def run(self, stop_event: threading.Event, finish_callback: callable = None):
        # threading.Thread(target=self.execute, args=(finish_callback,)).start()
        self.execute(stop_event, finish_callback)

    def execute(self, stop_event: threading.Event, finish_callback: callable = None):
        """
        Execute actions at 100Hz, and check for stop events frequently.
        """
        for action, duration in self.actions:
            if stop_event.is_set():
                print_t("[Go2] Action stopped.")
                break

            # print_t(f"[Go2] Executing action: {action.__name__ if hasattr(action, '__name__') else action} for {duration:.2f}s")
            action()

            start_time = time.time()
            while time.time() - start_time < duration:
                if stop_event.is_set():
                    break
                time.sleep(0.01)
        if finish_callback:
            finish_callback()
        
class Go2StateEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, RobotPosture):
            return obj.value
        return super().default(obj)

def go2action(feature_str = None):
    """
    Decorator to mark a function as an action that can be executed.
    This will create a Go2Action instance and execute it.
    
    Usage:
        @go2action("feature1,feature2")  # With features
        def my_action(self):
            pass
            
        @go2action  # Without features (no parentheses)
        def my_action(self):
            pass
            
        @go2action()  # Without features (with parentheses)
        def my_action(self):
            pass
    """
    
    # Handle the case where go2action is used without parentheses
    if callable(feature_str):
        # feature_str is actually the function being decorated
        func = feature_str
        features = []
        
        @wraps(func)
        def wrapper(self: "Go2Wrapper", *args, **kwargs):
            print_t(f">>> [Go2] Action: {func.__name__}, args: {args}, kwargs: {kwargs}")
            if "sit_stand" not in features and self.observation.posture == RobotPosture.LYING:
                self._go2_command("stand_up")

            self.observation.posture = RobotPosture.MOVING
            try:
                print_t(f">>> [Go2] Action {func.__name__} started, executing with args: {args}, kwargs: {kwargs}")
                result = func(self, *args, **kwargs)
            finally:
                print_t(f">>> [Go2] Action {func.__name__} completed")

            self.observation.posture = RobotPosture.STANDING
            return result
        
        return wrapper
    
    # Handle the case where go2action is used with parentheses but no arguments
    if feature_str is None:
        feature_str = ""
    
    # Normal case with parentheses and arguments
    def decorator(func):
        features = [f.strip() for f in feature_str.split(",")] if feature_str else []
        
        @wraps(func)
        def wrapper(self: "Go2Wrapper", *args, **kwargs):
            print_t(f">>> [Go2] Executing action: {func.__name__}, features: {features}")
            if "sit_stand" not in features and self.observation.posture == RobotPosture.LYING:
                self._go2_command("stand_up")

            self.observation.posture = RobotPosture.MOVING
            try:
                print_t(f">>> [Go2] Action {func.__name__} started, executing with args: {args}, kwargs: {kwargs}")
                result = func(self, *args, **kwargs)
            finally:
                print_t(f">>> [Go2] Action {func.__name__} completed, releasing lock.")

            self.observation.posture = RobotPosture.STANDING
            return result
        
        return wrapper
    
    return decorator

class Go2Wrapper(RobotWrapper):
    def __init__(self, robot_info: RobotInfo):
        self.init_ros_node()
        super().__init__(robot_info, Go2Observation(robot_info=robot_info, node=self.node))

        self.active_program = None
        self.action_lock = threading.RLock()

        # Connect to the robot control API
        extra = self.robot_info.extra
        if "url" not in extra:
            raise ValueError("Control url must be provided in extra")
        self.robot_url = extra["url"]
        try:
            response = requests.get(self.robot_url)
            if response.status_code != 200:
                raise RuntimeError(f"[Go2] Robot init connection failed: {response.status_code}, {response.text}")
            else:
                print_t(f"[Go2] {response.text}")
        except Exception as e:
            if not DEBUG_MODE:
                print_t(f"[Go2] Error connecting to robot at {self.robot_url}: {e}")
                raise RuntimeError(f"[Go2] Failed to connect to robot at {self.robot_url}")

        print_t(self.registry.get_skill_list())
        
        self.command_queue = queue.Queue()
        self.command_thread = threading.Thread(target=self.command_sender)

        self.spin_thread = threading.Thread(target=rclpy.spin, args=(self.node,))
        
        self.pid_yaw = PID(10.0, 0.0, 0.0, 10.0, 10.0, 0.5, 2.0)
        self.pid_x = PID(2.0, 1.0, 0.1, 10.0, 10.0, 0.5, 2.0)
        self.pid_y = PID(2.0, 1.0, 0.1, 10.0, 10.0, 0.5, 2.0)

    def init_ros_node(self):
        rclpy.init()
        self.node = rclpy.create_node('go2')

        # Create a subscription to the cmd_vel topic, working with Nav2
        self.node.create_subscription(
            Twist,
            '/cmd_vel',
            self._cmd_vel_callback,
            10
        )

        self.navigate_client = ActionClient(self.node, NavigateToPose, 'navigate_to_pose')

    def _cmd_vel_callback(self, msg: Twist):
        if msg.linear.x == 0.0 and msg.linear.y == 0.0 and msg.angular.z == 0.0:
            return
        control = {
            'vx': msg.linear.x,
            'vy': msg.linear.y,
            'vyaw': msg.angular.z
        }

        self._go2_command("nav", **control)

    # TODO: fix this
    def append_actions(self, actions: str):
        """
        Appends an action to the execution queue.
        """
        print_t(f"[Go2] Appending actions: {actions}, active program: {self.active_program.statement.to_string() if self.active_program else None}")
        actions = actions.strip().split(';')
        for action in actions:
            action = action.strip()
            if not action:
                continue

            if action == "continue()":
                continue
            elif action == "stop()":
                self.stop_action()
                continue
            # self.function_queue.put(action)
            print_t(f"[Go2] Executing action: {action}")

    # def worker(self):
    #     while self.running:
    #         try:
    #             action = self.function_queue.get(0.1)
    #             print_t(f"[Go2] Executing action: {action}")
    #             # self.active_program = MiniSpecProgram(self, None)
    #             # self.active_program.parse(action)
    #             # self.active_program.eval()
    #             self.registry.execute(action)
    #         except queue.Empty:
    #             pass

    @overrides
    def _start(self) -> bool:
        self.spin_thread.start()
        self.command_thread.start()
        print_t("[Go2] Robot is ready.")
        return True

    @overrides
    def _stop(self):
        self.command_thread.join()
        if self.spin_thread is not None:
            rclpy.shutdown()
            self.spin_thread.join()

    @overrides
    def _resume_action(self):
        pass

    @overrides
    def _pause_action(self):
        print_t("[Go2] Pausing action...")
        self._go2_command("stop")
        time.sleep(0.1)
    
    @overrides
    def _stop_action(self):
        print_t("[Go2] Stopping action...")
        self._go2_command("stop")
        time.sleep(0.1)

    def _go2_command(self, command: str, **args):
        # print_t(f"[Go2] Sending command: {command}, args: {args}")
        control = {
            "command": command,
            **args
        }

        match command:
            case "stand_up":
                self.observation.posture = RobotPosture.STANDING
            case "stand_down":
                self.observation.posture = RobotPosture.LYING
            case _:
                pass

        self.command_queue.put(control)

    def command_sender(self):
        send_request = lambda control: (
            # print(f"Sending control: {control}") or 
            requests.post(self.robot_url + '/control', json=control, 
                        headers={"Content-Type": "application/json"}, timeout=0.5)
        )
        current_euler = np.array([0.0, 0.0, 0.0])
        control = {"command": "stop"}
        last_euler_sent = 0.0
        while self.running:
            try:
                control = self.command_queue.get(timeout=0.05)
                if control["command"] == "stop":
                    current_euler[:] = 0.0
                    last_euler_sent = 0.0
                elif control["command"] == "nav" \
                        and (control.get("vx", 0.0) == 0.0
                        and control.get("vy", 0.0) == 0.0
                        and control.get("vyaw", 0.0) == 0.0
                        and (current_euler == 0.0).all()):
                    control = {"command": "stop"}
                elif control["command"] == "euler" \
                        and (control.get("roll", 0.0) == 0.0
                        and control.get("pitch", 0.0) == 0.0
                        and control.get("yaw", 0.0) == 0.0):
                    current_euler[:] = 0.0
                    control = {"command": "stop"}
            except queue.Empty:
                # if control["command"] != "euler":
                #     control = {"command": "stop"}

                # keep both euler and linear velocity
                pass

            if control["command"] == "euler":
                current_euler[0] = control.get("roll", 0.0)
                current_euler[1] = control.get("pitch", 0.0)
                current_euler[2] = control.get("yaw", 0.0)

                try:
                    result = send_request({
                        "command": "euler",
                        "roll": float(current_euler[0]),
                        "pitch": float(current_euler[1]),
                        "yaw": float(current_euler[2]),
                    })
                    result = send_request({'command': 'balanced_stand'})
                    if result.status_code != 200:
                        print_t(f"[Go2] Euler command failed: {result.status_code}, {result.text}")
                except requests.RequestException as e:
                    print_t(f"[Go2] Euler command failed: {e}")

                last_euler_sent = time.time()
            else:
                try:
                    result = send_request(control)
                    if result.status_code != 200:
                        print_t(f"[Go2] Command failed: {result.status_code}, {result.text}")
                except requests.RequestException as e:
                    if not DEBUG_MODE:
                        print_t(f"[Go2] Command send failed: {e}")

            # Periodic euler resend if active
            if (current_euler != 0.0).any() and (time.time() - last_euler_sent > 0.5):
                try:
                    result = send_request({
                        "command": "euler",
                        "roll": float(current_euler[0]),
                        "pitch": float(current_euler[1]),
                        "yaw": float(current_euler[2]),
                    })
                    if result.status_code != 200:
                        print_t(f"[Go2] Periodic euler resend failed: {result.status_code}, {result.text}")
                except requests.RequestException as e:
                    print_t(f"[Go2] Periodic euler resend failed: {e}")
                last_euler_sent = time.time()
        print_t("[Go2] Command sender stopped.")

    @robot_skill("stand_up", description="Make the robot stand up.", subsystem=SubSystem.MOVEMENT)
    @go2action("sit_stand")
    def stand_up(self):
        self._go2_command("stand_up")

    @robot_skill("sit_down", description="Make the robot sit down.", subsystem=SubSystem.MOVEMENT)
    @go2action("sit_stand")
    def sit_down(self):
        self._go2_command("stand_down")

    @robot_skill("orienting", description="Orient the robot's head to an object.", subsystem=SubSystem.MOVEMENT)
    @go2action
    def orienting(self, object: str, stop_event: threading.Event) -> bool:
        for _ in range(1):
            info = self.get_obj_info(object, True)
            if info is None:
                return False
            if (info.x - 0.5) < 0.1:
                return True

            self._rotate((0.5 - info.x) * 80, None, stop_event)

    @robot_skill("look_object", description="Look at a specific object", subsystem=SubSystem.MOVEMENT)
    @go2action
    def look_object(self, object: str, pause_event: threading.Event, stop_event: threading.Event) -> bool:
        body_pitch = self.observation.orientation[1]
        body_yaw = 0.0

        begin = time.time()
        while not stop_event.is_set():
            if pause_event.is_set():
                time.sleep(0.1)
                continue

            if time.time() - begin > 10.0:
                print_t("[Go2] Look object timeout.")
                break

            start_time = time.time()
            info = self.get_obj_info(object, True)

            if info is None:
                return False

            dx = 0.5 - info.cx
            dy = info.cy - 0.5

            # Dead zone in x-direction
            if abs(dx) > 0.05:
                body_yaw += dx / 4.0
            # Dead zone in y-direction (optional: adjust based on your use case)
            if abs(dy) > 0.05:
                body_pitch += dy / 6.0
                body_pitch = max(-0.75, min(0.75, body_pitch))

            # If yaw exceeds limits, rotate and skip the Euler update
            if abs(body_yaw) > 0.5:
                self._rotate(body_yaw * 180.0 / math.pi / 4, pause_event, stop_event)
                body_yaw = 0
                continue

            self._go2_command("euler", roll=0, pitch=round(float(body_pitch), 3), yaw=round(float(body_yaw), 3))
            time.sleep(max(0, 0.1 - (time.time() - start_time)))
        self._go2_command("euler", roll=0, pitch=0, yaw=0)
        return True

    # @go2action("require_standing, trigger_movement")
    # @overrides
    def _move(self, dx: float, dy: float, stop_event: threading.Event) -> bool:
        """
        Moves the robot by the specified distance in the x (forward/backward) and y (left/right) directions.
        """
        initial_x = self.observation.position[0]
        initial_y = self.observation.position[1]
        initial_yaw = self.observation.orientation[2]

        target_x = initial_x + dx * math.cos(initial_yaw) - dy * math.sin(initial_yaw)
        target_y = initial_y + dx * math.sin(initial_yaw) + dy * math.cos(initial_yaw)

        while not stop_event.is_set():
            start_time = time.time()
            current_x = self.observation.position[0]
            current_y = self.observation.position[1]
            current_yaw = self.observation.orientation[2]

            # Compute error in world frame
            error_world_x = target_x - current_x
            error_world_y = target_y - current_y

            # Convert error into body frame
            error_body_x = error_world_x * math.cos(current_yaw) + error_world_y * math.sin(current_yaw)
            error_body_y = -error_world_x * math.sin(current_yaw) + error_world_y * math.cos(current_yaw)

            if math.hypot(error_body_x, error_body_y) < 0.08:
                break

            vx = self.pid_x.update(error_body_x)
            vy = self.pid_y.update(error_body_y)

            self._go2_command("nav", vx=round(float(vx), 3), vy=round(float(vy), 3), vyaw=0.0)
            time.sleep(max(0, 0.1 - (time.time() - start_time)))
        self._go2_command("nav", vx=0.0, vy=0.0, vyaw=0.0)
        return True
    
    @robot_skill("move_forward", description="Move forward by a certain distance (m)", subsystem=SubSystem.MOVEMENT)
    @go2action
    def move_forward(self, distance: float, stop_event: threading.Event) -> bool:
        return self._move(distance, 0.0, stop_event)

    @robot_skill("move_back", description="Move back by a certain distance (m)", subsystem=SubSystem.MOVEMENT)
    @go2action
    def move_back(self, distance: float, stop_event: threading.Event) -> bool:
        return self._move(-distance, 0.0, stop_event)
    
    @robot_skill("move_left", description="Move left by a certain distance (m)", subsystem=SubSystem.MOVEMENT)
    @go2action
    def move_left(self, distance: float, stop_event: threading.Event) -> bool:
        return self._move(0.0, distance, stop_event)

    @robot_skill("move_right", description="Move right by a certain distance (m)", subsystem=SubSystem.MOVEMENT)
    @go2action
    def move_right(self, distance: float, stop_event: threading.Event) -> bool:
        return self._move(0.0, -distance, stop_event)

    @robot_skill("turn_left", description="Rotate counter-clockwise by a certain angle (degrees)", subsystem=SubSystem.MOVEMENT)
    @go2action
    def turn_left(self, deg: float, pause_event: threading.Event, stop_event: threading.Event) -> bool:
        return self._rotate(deg, pause_event, stop_event)

    @robot_skill("turn_right", description="Rotate clockwise by a certain angle (degrees)", subsystem=SubSystem.MOVEMENT)
    @go2action
    def turn_right(self, deg: float, pause_event: threading.Event, stop_event: threading.Event) -> bool:
        return self._rotate(-deg, pause_event, stop_event)

    def _rotate(self, deg: float, pause_event: Optional[threading.Event], stop_event: threading.Event) -> bool:
        """
        Rotates the robot by the specified angle in degrees.
        """
        print_t(f"-> Rotate by {deg} degrees")
        initial_yaw = self.observation.orientation[2]
        delta_rad = math.radians(deg)

        accumulated_angle = 0.0
        previous_yaw = initial_yaw

        while True:
            if stop_event.is_set():
                print_t("-> Rotation stopped by stop_event")
                return False

            if pause_event and pause_event.is_set():
                time.sleep(0.1)
                continue

            cycle_start_time = time.time()
            current_yaw = self.observation.orientation[2]
            yaw_diff = current_yaw - previous_yaw

            # Normalize yaw difference to the range [-pi, pi]
            if yaw_diff > math.pi:
                yaw_diff -= 2 * math.pi
            elif yaw_diff < -math.pi:
                yaw_diff += 2 * math.pi

            accumulated_angle += yaw_diff
            previous_yaw = current_yaw

            remaining_angle = delta_rad - accumulated_angle

            # print_t(f"-> Remaining angle: {math.degrees(remaining_angle):.2f} degrees, accumulated: {math.degrees(accumulated_angle):.2f} degrees")
            if abs(remaining_angle) < 0.01 or delta_rad * remaining_angle < 0:
                # If the remaining angle is small enough or we have overshot the target
                break

            vyaw = self.pid_yaw.update(remaining_angle)
            # print_t(f"-> vyaw: {vyaw:.2f} rad/s")
            self._go2_command("nav", vx=0.0, vy=0.0, vyaw=round(float(vyaw), 3))

            time.sleep(max(0, 0.1 - (time.time() - cycle_start_time)))
        self._go2_command("nav", vx=0.0, vy=0.0, vyaw=0.0)
        time.sleep(0.3)
        return True
    
    @robot_skill("nav", description="Continous movement and rotation with speed. Max speed +-1m/s, max turning +-0.5 rad/s. Positive value to move forward and clockwise/turn left.")
    @go2action
    def nav(self, vx: float, vyaw: float) -> bool:
        return self._go2_command("nav", vx=round(float(vx), 3), vy=0.0, vyaw=round(float(vyaw), 3))
    
    @robot_skill("search", description="Rotate to find a specific object when it's not in current view.")
    @go2action
    def search(self, object: str, pause_event: threading.Event, stop_event: threading.Event) -> bool:
        for _ in range(12):
            if stop_event.is_set():
                return False
            if pause_event.is_set():
                time.sleep(0.1)
                continue
            if self.get_obj_info(object) is not None:
                return True
            self._rotate(30, pause_event, stop_event)
        return False

    @robot_skill("follow", description="Follow a specific object.", subsystem=SubSystem.MOVEMENT)
    @go2action
    def follow(self, object: str, pause_event: threading.Event, stop_event: threading.Event) -> bool:
        last_seen_cx = 0.5
        current_pitch = 0.2
        self._go2_command("euler", roll=0, pitch=round(float(current_pitch), 2), yaw=0)
        start_time = time.time()
        while not stop_event.is_set():
            if pause_event.is_set():
                print_t("[Go2] Follow paused...")
                pause_event.wait()
                print_t("[Go2] Follow resumed...")

            if time.time() - start_time > 30.0:
                return False
            
            cycle_start_time = time.time()
            info = self.get_obj_info(object, True)
            if info is not None:
                last_seen_cx = info.cx
                if abs(last_seen_cx - 0.5) < 0.1:
                    vyaw = 0.0
                else:
                    vyaw = (0.5 - last_seen_cx) * 2.0
                    
                if info.depth > 1.5:
                    vx = min(1.0, info.depth / 3.0)
                elif info.depth < 0.8:
                    vx = -0.5
                else:
                    vx = 0.0

                if abs(info.cy - 0.5) > 0.2:
                    current_pitch += (info.cy - 0.5) / 3.0
                    self._go2_command("euler", roll=0, pitch=round(float(current_pitch), 2), yaw=0)
                    time.sleep(0.2)
                self._go2_command("nav", vx=round(float(vx), 3), vy=0.0, vyaw=round(float(vyaw), 3))
            else:
                current_pitch = 0.2
                self._go2_command("euler", roll=0, pitch=round(float(current_pitch), 2), yaw=0)
                if last_seen_cx - 0.5 < 0.0:
                    self._rotate(30, pause_event, stop_event)
                else:
                    self._rotate(-30, pause_event, stop_event)
            time.sleep(max(0, 0.1 - (time.time() - cycle_start_time)))
        self._go2_command("stop")
        return True

    @robot_skill("nod", description="Nod the robot's head.", subsystem=SubSystem.MOVEMENT)
    @go2action
    def nod(self, stop_event: threading.Event) -> bool:
        actions = []
        for _ in range(2):
            actions.append((lambda: self._go2_command("euler", roll=0, pitch=-0.4, yaw=0), 0.3))
            actions.append((lambda: self._go2_command("euler", roll=0, pitch=0.1, yaw=0), 0.3))
        go2_action = Go2Action(actions)
        go2_action.run(stop_event, lambda: self._go2_command("euler", roll=0, pitch=0, yaw=0))
        return True

    @robot_skill("wiggle", description="Wiggle the robot's body.", subsystem=SubSystem.MOVEMENT)
    @go2action
    def wiggle(self, stop_event: threading.Event) -> bool:
        actions = []
        for _ in range(2):
            actions.append((lambda: self._go2_command("euler", roll=0.6, pitch=0.0, yaw=0), 0.5))
            actions.append((lambda: self._go2_command("euler", roll=-0.6, pitch=0.0, yaw=0), 0.5))
        go2_action = Go2Action(actions)
        go2_action.run(stop_event, lambda: self._go2_command("euler", roll=0, pitch=0, yaw=0))
        return True

    @robot_skill("wagging", description="Wagging the robot's tail.", subsystem=SubSystem.MOVEMENT)
    @go2action
    def wagging(self, stop_event: threading.Event) -> bool:
        actions = []
        for _ in range(2):
            actions.append((lambda: self._go2_command("euler", roll=0.0, pitch=0.0, yaw=0.3), 0.5))
            actions.append((lambda: self._go2_command("euler", roll=0.0, pitch=0.0, yaw=-0.3), 0.5))
        go2_action = Go2Action(actions)
        go2_action.run(stop_event, lambda: self._go2_command("euler", roll=0, pitch=0, yaw=0))
        return True

    @robot_skill("stretch", description="Stretch the robot's limbs.", subsystem=SubSystem.MOVEMENT)
    @go2action
    def stretch(self) -> bool:
        self._go2_command("stretch")
        time.sleep(2.0)
        return True

    @robot_skill("speak", description="Make the robot speak.", subsystem=SubSystem.SOUND)
    @go2action
    def speak(self, text: str) -> bool:
        # self._go2_command("speak", text=text)
        print_t(f">>> Robot says: {text}")
        return True

    @robot_skill("look_up", description="Look up by adjusting the robot's head pitch.", subsystem=SubSystem.MOVEMENT)
    @go2action
    def look_up(self, stop_event: threading.Event) -> bool:
        actions = [
            (lambda: self._go2_command("euler", roll=0, pitch=-0.4, yaw=0), 0.2)
            for _ in range(3)
        ]
        go2_action = Go2Action(actions)
        go2_action.run(stop_event, lambda: self._go2_command("euler", roll=0, pitch=0, yaw=0))
        print_t("-> Look up end")
        return True

    @robot_skill("goto_waypoint", description="Go to a specific waypoint", subsystem=SubSystem.MOVEMENT)
    @go2action
    def goto_waypoint(self, id: int, pause_event: threading.Event, stop_event: threading.Event) -> bool:
        wp = self.observation.slam_map.get_waypoint(id)
        if wp is None:
            print_t(f"Waypoint {id} not found")
            return False
        return self._goto(wp.x, wp.y, 30.0, pause_event=pause_event, stop_event=stop_event)

    def _goto(self, x: float, y: float, timeout_sec: float, pause_event: threading.Event, stop_event: threading.Event) -> bool:
        print(f"-> Go to ({x}, {y})")

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.node.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = x
        goal_msg.pose.pose.position.y = y
        goal_msg.pose.pose.orientation.z = 0.0
        goal_msg.pose.pose.orientation.w = 1.0

        self.navigate_client.wait_for_server()

        done_event = threading.Event()
        result_status = {"status": None}
        goal_handle_container = {}

        def goal_response_callback(future):
            goal_handle = future.result()
            if not goal_handle.accepted:
                print_t("Navigation: Goal rejected")
                result_status["status"] = False
                done_event.set()
                return

            goal_handle_container["handle"] = goal_handle  # Save for timeout/cancel
            result_future = goal_handle.get_result_async()

            def result_callback(result_future):
                result = result_future.result()
                goal_id = goal_handle.goal_id
                if result.status == GoalStatus.STATUS_SUCCEEDED:  # Use enum for clarity
                    print_t("Navigation: Goal succeeded")
                    result_status["status"] = True
                else:
                    print_t(f"Navigation: Goal failed with status {result.status}")
                    result_status["status"] = False
                done_event.set()

            result_future.add_done_callback(result_callback)

        send_goal_future = self.navigate_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(goal_response_callback)

        start_time = time.time()
        while not done_event.is_set():
            if stop_event.is_set() or time.time() - start_time > timeout_sec:
                print_t("Navigation: Stopped")
                break
            if pause_event.is_set():
                time.sleep(0.1)
                continue
            time.sleep(0.01)

        if not done_event.is_set():
            goal_handle = goal_handle_container.get("handle")
            if goal_handle and goal_handle.accepted:
                goal_handle.cancel_goal_async()
            result_status["status"] = False
        return result_status["status"]