import time, os, math
import numpy as np
import threading, requests
from overrides import overrides
from PIL import Image
import cv2
import json
import queue
from enum import Enum
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import Image as ROSImage
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage
from nav_msgs.msg import OccupancyGrid
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus

from typego.robot_wrapper import RobotWrapper, RobotObservation, RobotPosture
from typego.robot_info import RobotInfo
from typego.minispec_interpreter import MiniSpecProgram
from typego.yolo_client import YoloClient
from typego.skillset import SkillSet, SkillArg, SkillSetLevel
from typego.utils import quaternion_to_rpy, print_t, ImageRecover

from typego_interface.msg import WayPointArray

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

GO2_CAM_K = np.array([
    [818.18507419, 0.0, 637.94628188],
    [0.0, 815.32431463, 338.3480119],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

GO2_CAM_D = np.array([[-0.07203219],
                      [-0.05228525],
                      [ 0.05415833],
                      [-0.02288355]], dtype=np.float32)

def make_transform(translation, quaternion):
    T = np.eye(4)
    T[:3, :3] = R.from_quat(quaternion).as_matrix()
    T[:3, 3] = translation
    return T

class Go2Observation(RobotObservation):
    def __init__(self, robot_info: RobotInfo, node: Node, rate: int = 10):
        super().__init__(robot_info, rate)
        self.yolo_client = YoloClient(robot_info)
        self.image_receover = ImageRecover(GO2_CAM_K, GO2_CAM_D)
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
            10
        )

        # Subscribe to /tf
        node.create_subscription(
            TFMessage,
            '/tf',
            self._tf_callback,
            10
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
            10
        )

        self.closest_object: tuple[float, float] = (float('inf'), 0.0)  # (distance, angle)

    def _image_callback(self, msg: ROSImage):
        # Convert ROS Image message to OpenCV image
        cv_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        # Undistort the image
        frame = self.image_receover.process(cv_image)
        self.image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def _tf_callback(self, msg: TFMessage):
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
        # If the closest object is within 0.4m and it's directly in front of the robot
        distance, angle = self.closest_object
        return distance < 0.4 and abs(angle) < math.radians(60)
        
    @overrides
    def _start(self):
        return
    
    @overrides
    def _stop(self):
        return
        
    @overrides
    async def process_image(self, image: Image.Image):
        await self.yolo_client.detect(image)
    
    @overrides
    def fetch_processed_result(self) -> tuple[Image.Image, list] | None:
        return self.yolo_client.latest_result

class Go2Action:
    event: threading.Event = None
    def __init__(self, actions: list[tuple[callable, float]]):
        self.actions = actions  # List of tuples (function, duration)

        if self.event is None:
            raise ValueError("Go2Action event is not initialized. Please call Go2Wrapper.start() first.")
    
    def execute(self):
        """
        Execute actions at 10Hz, and check for stop events frequently.
        """
        for action, duration in self.actions:
            if self.event.is_set():
                print_t("[Go2] Action stopped.")
                return

            # print_t(f"[Go2] Executing action: {action.__name__ if hasattr(action, '__name__') else action} for {duration:.2f}s")
            action()

            start_time = time.time()
            while time.time() - start_time < duration:
                if self.event.is_set():
                    print_t("[Go2] Action interrupted during execution.")
                    return
                time.sleep(0.01)

        # print_t("[Go2] Action execution completed.")
        
class Go2StateEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, RobotPosture):
            return obj.value
        return super().default(obj)

class Go2Wrapper(RobotWrapper):
    def __init__(self, robot_info: RobotInfo, system_skill_func: list[callable]):
        self.init_ros_node()
        super().__init__(robot_info, Go2Observation(robot_info=robot_info, node=self.node), system_skill_func)

        self.running = True
        self.state = {
            "posture": RobotPosture.STANDING,
            "x": 0.0, "y": 0.0,
            "yaw": 0,
        }

        self.execution_queue = queue.Queue()
        self.active_program = None
        self.stop_action_event = threading.Event()
        Go2Action.event = self.stop_action_event

        extra = self.robot_info.extra
        if "url" not in extra:
            raise ValueError("Control url must be provided in extra")
        self.robot_url = extra["url"]
        response = requests.get(self.robot_url)
        if response.status_code != 200:
            raise RuntimeError(f"[Go2] Failed to connect to robot at {self.robot_url}")
        else:
            print_t(f"[Go2] {response.text}")

        self.ll_skillset.add_low_level_skill("stand_up", lambda: self._action('stand_up'), "Stand up")
        self.ll_skillset.add_low_level_skill("lie_down", lambda: self._action('stand_down'), "Stand down")
        # self.ll_skillset.add_low_level_skill("goto", self.goto, "Go to a specific position (x, y) in m", args=[SkillArg("x", float), SkillArg("y", float)])
        self.ll_skillset.add_low_level_skill("goto_waypoint", self.goto_waypoint, "Go to a way point", args=[SkillArg("id", int)])
        self.ll_skillset.add_low_level_skill("stop", self.stop_action, "Stop current action")
        self.ll_skillset.add_low_level_skill("look_object", self.look_object, "Look at an object", args=[SkillArg("object_name", str)])
        self.ll_skillset.add_low_level_skill("nod", self.nod, "Nod the robot's head")
        self.ll_skillset.add_low_level_skill("look_up", self.look_up, "Look up")

        high_level_skills = [
            {
                "name": "scan",
                "definition": "{8{?is_visible($1){->True}rotate(45)}->False}",
                "description": "Rotate to find a specific object $1 when it's *not* in current view",
            },
            {
                "name": "scan_description",
                "definition": "{8{_1=probe($1);?_1!=False{->_1}rotate(45)}->False}",
                "description": "Rotate to find an abstract object $1 when it's *not* in current view",
            },
            {
                "name": "orienting",
                "definition": "{_1=object_x($1);rotate((0.5-_1)*80)}",
                "description": "Rotate to align with object $1",
            },
            {
                "name": "goto_object",
                "definition": "2{orienting($1);_1=object_distance($1)/3;{move(_1, 0)}}",
                "description": "Move to object $1 in the view (orienting then go forward)"
            }
        ]

        self.hl_skillset = SkillSet(SkillSetLevel.HIGH, self.ll_skillset)
        for skill in high_level_skills:
            self.hl_skillset.add_high_level_skill(skill['name'], skill['definition'], skill['description'])

        self.execution_thread = threading.Thread(target=self.worker)
        self.execution_thread.start()

    def init_ros_node(self):
        rclpy.init()
        self.node = rclpy.create_node('go2')
        # Subscribe to /camera/image_raw
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
            'command': 'nav',
            'vx': msg.linear.x,
            'vy': msg.linear.y,
            'vyaw': msg.angular.z
        }

        self._send_control(control)

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

            if action == "keep()":
                continue
            elif action == "stop()":
                self.stop_action()
                continue
            elif action == "done(True)":
                self.memory.end_inst(True)
                continue
            elif action == "done(False)":
                self.memory.end_inst(False)
                continue
            self.execution_queue.put(action)

    def worker(self):
        while self.running:
            if not self.execution_queue.empty():
                action = self.execution_queue.get()
                print_t(f"[Go2] Executing action: {action}")
                self.active_program = MiniSpecProgram(self, None)
                self.active_program.parse(action)
                self.active_program.eval()
            time.sleep(0.1)

    @overrides
    def start(self) -> bool:
        self.spin_thread = threading.Thread(target=rclpy.spin, args=(self.node,))
        self.spin_thread.start()
        self.observation.start()
        return True

    @overrides
    def stop(self):
        self.running = False
        self.execution_thread.join()
        self.observation.stop()
        if self.spin_thread is not None:
            rclpy.shutdown()
            self.spin_thread.join()

    @overrides
    def get_state(self) -> str:
        js = {
            "time": time.strftime("%H:%M:%S", time.localtime(time.time())),
            "posture": self.state["posture"].value,
            "waypoint_id": self.observation.slam_map.get_nearest_waypoint_id((self.observation.position[0], self.observation.position[1])),
        }
        return json.dumps(js)
    
    @overrides
    def get_posture(self) -> RobotPosture:
        return self.state["posture"]
    
    def _stand_check(self):
        if self.state["posture"] == RobotPosture.LYING:
            self._action("stand_up")
            self.state["posture"] = RobotPosture.STANDING

    def stop_action(self):
        print_t("[Go2] Stopping action...")
        self.memory.stop_action()
        self.stop_action_event.set()
        if self.active_program:
            self.active_program.stop()
        time.sleep(0.05)
        self.stop_action_event.clear()

    def look_object(self, object_name: str, timeout: float = 4.0) -> bool:
        self._stand_check()
        body_pitch = self.observation.orientation[1]
        body_yaw = 0

        start_time = time.time()
        while not self.stop_action_event.is_set() and time.time() - start_time < timeout:
            info = self.get_obj_info(object_name)

            if info is None:
                return False

            dx = 0.5 - info.x
            dy = info.y - 0.5

            # Dead zone in x-direction
            if abs(dx) > 0.05:
                body_yaw += dx / 4.0
            # Dead zone in y-direction (optional: adjust based on your use case)
            if abs(dy) > 0.05:
                body_pitch += dy / 6.0
                body_pitch = max(-0.75, min(0.75, body_pitch))

            # If yaw exceeds limits, rotate and skip the Euler update
            if abs(body_yaw) > 0.5:
                self.rotate(body_yaw * 180.0 / math.pi / 4)
                body_yaw = 0
                continue

            actions = [
                (lambda: self._action("euler", roll=0, pitch=body_pitch, yaw=body_yaw), 0.1)
            ]
            go2_action = Go2Action(actions)
            go2_action.execute()

        return True

    def _action(self, action: str, **args):
        control = {
            "command": action,
            "timeout": 3.0,
            **args
        }

        match action:
            case "stand_up":
                self.state["posture"] = RobotPosture.STANDING
            case "stand_down":
                self.state["posture"] = RobotPosture.LYING
            case _:
                pass

        self._send_control(control)

    def _send_control(self, control: dict):
        response = requests.post(
            self.robot_url + "control",
            json=control,
            headers={"Content-Type": "application/json"}
        )
        if response.status_code != 200:
            print_t(f"[Go2] Failed to send command: {response.text}")

    def _move(self, linear_x: float=0.0, linear_y: float=0.0, angular_z: float=0.0, duration: float=3.0):
        """
        Helper function to publish Twist messages for a specified duration.
        """
        control = { "timeout": duration }
        if linear_x != 0.0 or linear_y != 0.0:
            control["command"] = "move"
            control["dx"] = linear_x
            control["dy"] = linear_y
            control["body_frame"] = True
        elif angular_z != 0.0:
            control["command"] = "rotate"
            control["delta_rad"] = angular_z
        self._send_control(control)

    @overrides
    def move(self, dx: float, dy: float) -> bool:
        """
        Moves the robot by the specified distance in the x (forward/backward) and y (left/right) directions.
        """
        print(f"-> Move by ({dx}, {dy}) m")
        self._stand_check()

        self.state["posture"] = RobotPosture.MOVING
        self._move(linear_x=dx, linear_y=dy, duration=5.0)
        self.state["posture"] = RobotPosture.STANDING
        return True

    @overrides
    def rotate(self, deg: float) -> bool:
        """
        Rotates the robot by the specified angle in degrees.
        """
        print_t(f"-> Rotate by {deg} degrees")
        self._stand_check()
        self.state["posture"] = RobotPosture.MOVING
        self._move(angular_z=math.radians(deg), duration=5.0)
        self.state["posture"] = RobotPosture.STANDING
        return True
    
    def nod(self) -> bool:
        """
        Nods the robot's head.
        """
        print("-> Nod")
        self._stand_check()
        actions = []
        for _ in range(2):  # 2 up/down cycles = 4 total motions
            actions.append((lambda: self._action("euler", roll=0, pitch=-0.2, yaw=0), 0.5))
            actions.append((lambda: self._action("euler", roll=0, pitch=0.0, yaw=0), 0.5))
        go2_action = Go2Action(actions)
        go2_action.execute()
        self._action("euler", roll=0, pitch=0.0, yaw=0)
        return True
    
    def look_up(self) -> bool:
        """
        Looks up by adjusting the robot's head pitch.
        """
        print_t("-> Look up")
        self._stand_check()
        actions = [
            (lambda: self._action("euler", roll=0, pitch=-0.4, yaw=0), 0.2)
            for _ in range(15)  # 3 up/down cycles
        ]
        go2_action = Go2Action(actions)
        go2_action.execute()
        print_t("-> Look up end")
        return True
    
    def goto_waypoint(self, id: int) -> bool:
        print(f"-> Go to waypoint {id}")
        self._stand_check()
        wp = self.observation.slam_map.get_waypoint(id)
        if wp is None:
            print_t(f"Waypoint {id} not found")
            return False
        return self.goto(wp.x, wp.y)
    
    def goto(self, x: float, y: float, timeout_sec: float = 30.0) -> bool:
        print(f"-> Go to ({x}, {y})")
        self._stand_check()
        self.state["posture"] = RobotPosture.MOVING

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
            if self.stop_action_event.is_set() or time.time() - start_time > timeout_sec:
                print_t("Navigation: Stopped")
                break
            time.sleep(0.1)

        if not done_event.is_set():
            goal_handle = goal_handle_container.get("handle")
            if goal_handle and goal_handle.accepted:
                goal_handle.cancel_goal_async()
            result_status["status"] = False

        self.state["posture"] = RobotPosture.STANDING
        return result_status["status"]