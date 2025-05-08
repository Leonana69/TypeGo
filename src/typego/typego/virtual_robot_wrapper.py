import cv2, time
import threading
from PIL import Image
from overrides import overrides
import json

from typego.skillset import SkillSet, SkillSetLevel
from typego.robot_wrapper import RobotWrapper, RobotObservation
from typego.yolo_client import YoloClient
from typego.robot_info import RobotInfo

SKILL_EXECUTION_TIME = 2

class VirtualObservation(RobotObservation):
    def __init__(self, robot_info: RobotInfo, rate: int = 10):
        super().__init__(robot_info, rate)
        self.yolo_client = YoloClient(robot_info)

        if "capture" not in robot_info.extra:
            raise ValueError("Robot info must contain 'capture' key in extra, which is the camera index")

        self.cap: cv2.VideoCapture = None
        def _capture_spin():
            # must create the capture and read in the same thread
            self.cap = cv2.VideoCapture(int(self.robot_info.extra["capture"]))
            if not self.cap.isOpened():
                raise RuntimeError("Failed to open GStreamer pipeline")
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                # Convert the frame to RGB and store it in self._image
                self._image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cv2.waitKey(1)
        self.capture_thread = threading.Thread(target=_capture_spin)
    
    @overrides
    def _start(self):
        self.capture_thread.start()

    @overrides  
    def _stop(self):
        self.capture_thread.join()
        if self.cap is not None:
            self.cap.release()

    @overrides
    async def process_image(self, image: Image.Image):
        await self.yolo_client.detect(image)
    
    @overrides
    def fetch_processed_result(self) -> tuple[Image.Image, list]:
        return self.yolo_client.latest_result

from enum import Enum
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

class VirtualRobotWrapper(RobotWrapper):
    def __init__(self, robot_info: RobotInfo, system_skill_func: list[callable]):
        super().__init__(robot_info, VirtualObservation(robot_info), system_skill_func)

        self.state = {
            "posture": Go2Posture.STANDING,
            "x": 0.0,
            "y": 0.0,
            "yaw": 0.0,
        }

        # extra movement skills
        self.ll_skillset.add_low_level_skill("stand_up", lambda: None, "Stand up")
        self.ll_skillset.add_low_level_skill("lying_down", lambda: None, "Stand down")
        
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
    def move(self, dx: float, dy: float) -> bool:
        print(f"-> Move by ({dx}, {dy}) cm")
        self.state["posture"] = Go2Posture.MOVING
        time.sleep(SKILL_EXECUTION_TIME)
        # Update the location based on dx and dy
        self.state["x"] += dx
        self.state["y"] += dy
        self.state["posture"] = Go2Posture.STANDING
        return True

    @overrides
    def rotate(self, deg: float) -> bool:
        print(f"-> Rotate by {deg} degrees")
        self.state["posture"] = Go2Posture.MOVING
        time.sleep(SKILL_EXECUTION_TIME)
        # Update the yaw based on deg
        self.state["yaw"] += deg
        self.state["posture"] = Go2Posture.STANDING
        return True
    
    @overrides
    def get_state(self) -> str:
        return json.dumps(self.state)

    def _action(self, action: str):
        match action:
            case "stand_up":
                self.state["posture"] = Go2Posture.STANDING
            case "stand_down":
                self.state["posture"] = Go2Posture.LYING
            case _:
                pass