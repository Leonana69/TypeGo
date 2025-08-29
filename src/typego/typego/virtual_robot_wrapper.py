import cv2, time
import threading
from PIL import Image
from overrides import overrides
import json

from typego.robot_wrapper import RobotWrapper, RobotObservation
from typego.yolo_client import YoloClient
from typego.robot_info import RobotInfo
from typego.robot_wrapper import RobotPosture

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
            while not self._stop_evt.is_set():
                ret, frame = self.cap.read()
                if not ret:
                    continue
                # Convert the frame to RGB and store it in self._image
                self._image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cv2.waitKey(1)
            # while True:
            #     time.sleep(1)
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
    
    @overrides
    def obs(self) -> dict:
        return {
            "posture": self.posture
        }

class VirtualRobotWrapper(RobotWrapper):
    def __init__(self, robot_info: RobotInfo):
        super().__init__(robot_info, VirtualObservation(robot_info))

    @overrides
    def start(self) -> bool:
        self.observation.start()
        return True

    @overrides
    def stop(self):
        self.observation.stop()

    @overrides
    def move_forward(self, distance: float) -> bool:
        print(f"-> Move forward by {distance} cm")
        time.sleep(SKILL_EXECUTION_TIME)
        return True

    @overrides
    def move_back(self, distance: float) -> bool:
        print(f"-> Move back by {distance} cm")
        time.sleep(SKILL_EXECUTION_TIME)
        return True

    @overrides
    def move_left(self, distance: float) -> bool:
        print(f"-> Move left by {distance} cm")
        time.sleep(SKILL_EXECUTION_TIME)
        return True

    @overrides
    def move_right(self, distance: float) -> bool:
        print(f"-> Move right by {distance} cm")
        time.sleep(SKILL_EXECUTION_TIME)
        return True
    
    @overrides
    def turn_left(self, deg: float) -> bool:
        print(f"-> Turn left by {deg} degrees")
        time.sleep(SKILL_EXECUTION_TIME)
        return True

    @overrides
    def turn_right(self, deg: float) -> bool:
        print(f"-> Turn right by {deg} degrees")
        time.sleep(SKILL_EXECUTION_TIME)
        return True