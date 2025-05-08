import time, os, math
import numpy as np
import threading, requests
from overrides import overrides
from PIL import Image
import cv2
import json
from enum import Enum

from ..robot_wrapper import RobotWrapper, RobotObservation
from ..robot_info import RobotInfo
from ..yolo_client import YoloClient
from ..skillset import SkillSet, SkillArg, SkillSetLevel
from ..utils import quaternion_to_rpy, print_t, undistort_image

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

class Go2Observation(RobotObservation):
    def __init__(self, robot_info: RobotInfo, rate: int = 10):
        super().__init__(robot_info, rate)
        self.yolo_client = YoloClient(robot_info)
        self.init_custom_sdk()

    def init_custom_sdk(self):
        # Use gstreamer and OpenCV to read the video stream
        # You need to start the gstreamer pipeline on the robot, see platforms/README.md
        GSTREAMER_PIPELINE_STR = """
            udpsrc address=230.1.1.1 port=1720 multicast-iface=wlan0
            ! application/x-rtp, media=video, encoding-name=H264
            ! rtph264depay
            ! h264parse
            ! avdec_h264
            ! videoconvert
            ! video/x-raw, format=BGR
            ! appsink name=appsink emit-signals=true max-buffers=1 drop=true
        """
        self.gstreamer_cap: cv2.VideoCapture = None
        def _gstreamer_spin():
            # must create the capture and read in the same thread
            self.gstreamer_cap = cv2.VideoCapture(GSTREAMER_PIPELINE_STR, cv2.CAP_GSTREAMER)
            if not self.gstreamer_cap.isOpened():
                raise RuntimeError("Failed to open GStreamer pipeline")
            while self.running:
                ret, frame = self.gstreamer_cap.read()
                if not ret:
                    continue
                # Convert the frame to RGB and store it in self._image
                
                frame = undistort_image(frame, GO2_CAM_K, GO2_CAM_D)
                self.image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cv2.waitKey(1)
        self.gstreamer_thread = threading.Thread(target=_gstreamer_spin)

        # Initialize state reader
        def _state_reader():
            while self.running:
                # Read the state from the robot
                response = requests.get(self.robot_info.extra["url"] + "state")
                if response.status_code == 200:
                    data = response.json()
                    self.position = np.array([data['x'], data['y'], data['z']])
                    self.orientation = np.array(
                        [data['roll'], data['pitch'], data['yaw']]
                    )
                time.sleep(0.1)

        self.state_thread = threading.Thread(target=_state_reader)
        
    @overrides
    def _start(self):
        self.gstreamer_thread.start()
        self.state_thread.start()
    
    @overrides
    def _stop(self):
        self.gstreamer_thread.join()
        self.state_thread.join()
        if self.gstreamer_cap is not None:
            self.gstreamer_cap.release()
        
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
        