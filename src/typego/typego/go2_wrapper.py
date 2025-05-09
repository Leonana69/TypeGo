import time, os, math
import numpy as np
import threading, requests
from overrides import overrides
from PIL import Image
import cv2
import json
from enum import Enum
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROSImage
from tf2_msgs.msg import TFMessage
from nav_msgs.msg import OccupancyGrid

from typego.robot_wrapper import RobotWrapper, RobotObservation
from typego.robot_info import RobotInfo
from typego.yolo_client import YoloClient
from typego.skillset import SkillSet, SkillArg, SkillSetLevel
from typego.utils import quaternion_to_rpy, print_t, ImageRecover

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
    def __init__(self, robot_info: RobotInfo, rate: int = 10):
        super().__init__(robot_info, rate)
        self.yolo_client = YoloClient(robot_info)
        self.image_receover = ImageRecover(GO2_CAM_K, GO2_CAM_D)
        self.init_ros_obs()

        self.map2odom_translation = np.array([0.0, 0.0, 0.0])
        self.map2odom_rotation = np.array([0.0, 0.0, 0.0, 1.0])
        self.odom2robot_translation = np.array([0.0, 0.0, 0.0])
        self.odom2robot_rotation = np.array([0.0, 0.0, 0.0, 1.0])

    def init_ros_obs(self):
        rclpy.init()
        self.node = rclpy.create_node('go2_observation')
        # Subscribe to /camera/image_raw
        self.node.create_subscription(
            ROSImage,
            '/camera/image_raw',
            self._image_callback,
            10
        )

        # Subscribe to /tf
        self.node.create_subscription(
            TFMessage,
            '/tf',
            self._tf_callback,
            10
        )

        # Subscribe to /map
        self.node.create_subscription(
            OccupancyGrid,
            '/map',
            self._map_callback,
            10
        )

    def _image_callback(self, msg: ROSImage):
        # Convert ROS Image message to OpenCV image
        cv_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        # Undistort the image
        frame = self.image_receover.process(cv_image)
        # Convert to PIL Image and store it
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

    def _map_callback(self, msg: OccupancyGrid):
        width = msg.info.width
        height = msg.info.height
        data = np.array(msg.data, dtype=np.int8).reshape((height, width))

        # Convert OccupancyGrid to grayscale image
        map_data = np.zeros((height, width), dtype=np.uint8)
        map_data[data == 0] = 255      # Free space = white
        map_data[data == -1] = 128     # Unknown = gray
        map_data[data > 0] = 0         # Occupied = black
        map_data = cv2.flip(map_data, 0)

        # Convert to Colored map_data
        map_data = cv2.cvtColor(map_data, cv2.COLOR_GRAY2BGR)
        self.slam_map.update_map(map_data, width, height, (msg.info.origin.position.x, msg.info.origin.position.y), msg.info.resolution)
        
    @overrides
    def _start(self):
        self.spin_thread = threading.Thread(target=rclpy.spin, args=(self.node,))
        self.spin_thread.start()
    
    @overrides
    def _stop(self):
        rclpy.shutdown()
        if self.spin_thread is not None:
            self.spin_thread.join()
        
    @overrides
    async def process_image(self, image: Image.Image):
        await self.yolo_client.detect(image)
    
    @overrides
    def fetch_processed_result(self) -> tuple[Image.Image, list]:
        return self.yolo_client.latest_result

class Go2Posture(Enum):
    STANDING = "standing"
    LYING = "lying"
    MOVING = "moving"

    @staticmethod
    def from_string(s: str):
        if s == "standing":
            return Go2Posture.STANDING
        elif s == "lying":
            return Go2Posture.LYING
        elif s == "moving":
            return Go2Posture.MOVING
        else:
            raise ValueError(f"Unknown posture: {s}")

class Go2Wrapper(RobotWrapper):
    def __init__(self, robot_info: RobotInfo, system_skill_func: list[callable]):
        super().__init__(robot_info, Go2Observation(robot_info), system_skill_func)

        self.state = {
            "posture": Go2Posture.STANDING,
            "x": 0.0, "y": 0.0,
            "yaw": 0.0,
        }

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
        self.ll_skillset.add_low_level_skill("lying_down", lambda: self._action('stand_down'), "Stand down")

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
                "definition": "{_1=ox($1);rotate((0.5-_1)*80)}",
                "description": "Rotate to align with object $1",
            },
            {
                "name": "goto",
                "definition": "?orienting($1){move(80, 0)}",
                "description": "Move to object $1 in the view"
            }
        ]

        self.hl_skillset = SkillSet(SkillSetLevel.HIGH, self.ll_skillset)
        for skill in high_level_skills:
            self.hl_skillset.add_high_level_skill(skill['name'], skill['definition'], skill['description'])

    @overrides
    def start(self) -> bool:
        self.observation.start()
        return True

    @overrides
    def stop(self):
        self.observation.stop()

    @overrides
    def get_state(self) -> str:
        self.state["x"] = self.observation.position[0]
        self.state["y"] = self.observation.position[1]
        self.state["yaw"] = self.observation.orientation[2] * 180.0 / math.pi
        return json.dumps(self.state)

    def _action(self, action: str):
        control = {
            "command": action,
            "timeout": 3.0
        }

        match action:
            case "stand_up":
                self.state["posture"] = Go2Posture.STANDING
            case "stand_down":
                self.state["posture"] = Go2Posture.LYING
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
            print_t(f"[Go2] Failed to send command: {response.json()}")

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
        print(f"-> Move by ({dx}, {dy}) cm")
        self.state["posture"] = Go2Posture.MOVING
        # Convert distances from cm to meters
        dx_m = dx / 100.0
        dy_m = dy / 100.0

        # Perform the movement
        self._move(linear_x=dx_m, linear_y=dy_m, duration=5.0)
        self.state["posture"] = Go2Posture.STANDING
        return True

    @overrides
    def rotate(self, deg: float) -> bool:
        """
        Rotates the robot by the specified angle in degrees.
        """
        print(f"-> Rotate by {deg} degrees")
        self.state["posture"] = Go2Posture.MOVING
        self._move(angular_z=math.radians(deg), duration=5.0)
        self.state["posture"] = Go2Posture.STANDING
        return True
        