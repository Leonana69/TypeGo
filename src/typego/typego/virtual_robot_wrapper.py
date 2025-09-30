import cv2, time
import threading
from PIL import Image
from overrides import overrides
import json

from typego.robot_wrapper import RobotWrapper, RobotObservation, robot_skill
from typego.yolo_client import YoloClient
from typego.robot_info import RobotInfo
from typego.robot_wrapper import RobotPosture
from typego.method import make_find_object_method
from typego.skill_item import SubSystem
from typego.utils import print_t

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
            "t": time.strftime("%H:%M:%S", time.localtime(time.time())),
            "robot": {
                "posture": self.posture.name.lower(),
            },
            "perception": self.yolo_client.latest_result[1] if self.yolo_client.latest_result else [],
        }

class VirtualRobotWrapper(RobotWrapper):
    def __init__(self, robot_info: RobotInfo):
        super().__init__(robot_info, VirtualObservation(robot_info))
        print_t(self.registry.get_skill_list())

    @overrides
    def _start(self) -> bool:
        self.observation.posture = RobotPosture.STANDING
        return True

    @overrides
    def _stop(self):
        pass

    @overrides
    def _pause_action(self):
        pass

    @overrides
    def _resume_action(self):
        pass

    @overrides
    def _stop_action(self):
        pass

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
    
    @robot_skill("orienting", description="Orient the robot's head to an object.", subsystem=SubSystem.MOVEMENT)
    def orienting(self, object: str, stop_event: threading.Event) -> bool:
        print_t(f"-> Orienting head to {object}")
        start_time = time.time()
        while not stop_event.is_set() and time.time() - start_time < SKILL_EXECUTION_TIME:
            print_t(f"   ... orienting head to {object} ...")
            time.sleep(1)
        print_t(f"-> Finished orienting head to {object}")
        return True