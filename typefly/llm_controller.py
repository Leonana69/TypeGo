from PIL import Image
import queue, io, base64
from openai import Stream
from typing import Optional
import time

from .yolo_client import YoloClient
from .platforms.virtual_robot_wrapper import VirtualRobotWrapper
from .robot_wrapper import RobotWrapper
from .llm_planner import LLMPlanner
from .utils import print_t
from .minispec_interpreter import MiniSpecInterpreter
from .robot_info import RobotInfo

class LLMController():
    def __init__(self, robot_info: RobotInfo, message_queue: Optional[queue.Queue]=None):
        self.message_queue = message_queue

        self.planner = LLMPlanner()

        self.controller_func = [
            self.user_log,
            self.probe
        ]

        if robot_info.robot_type == "virtual":
            self.robot = VirtualRobotWrapper(robot_info, self.controller_func)
        elif robot_info.robot_type == "tello":
            from .platforms.tello_wrapper import TelloWrapper
            self.robot = TelloWrapper(robot_info, self.controller_func)
        elif robot_info.robot_type == "go2":
            from .platforms.go2_wrapper import Go2Wrapper
            self.robot = Go2Wrapper(robot_info, self.controller_func)
        elif robot_info.robot_type == "pod":
            from .platforms.pod_wrapper import PodWrapper
            self.robot = PodWrapper(robot_info, self.controller_func)
        
        self.planner.set_robot(self.robot)
        self.current_plan = None
        self.execution_history = None

    def user_log(self, content: str | Image.Image) -> bool:
        if isinstance(content, Image.Image):
            buffer = io.BytesIO()
            content.save(buffer, format="JPEG")
            self._send_message(f'<img src="data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode("utf-8")}" />')
        else:
            text = content.strip('\'')
            self._send_message(f'[LOG] {text}')
            print_t(f'[LOG] {text}')
        return True

    def probe(self, query: str) -> str:
        return self.planner.probe(query)

    def _send_message(self, message: str):
        if self.message_queue is not None:
            self.message_queue.put(message)

    def start_controller(self):
        self.robot.start()
        
    def stop_controller(self):
        self.robot.stop()

    def fetch_robot_observation(self, overlay: bool=False) -> Optional[Image.Image]:
        obs = self.robot.observation
        if not obs or not obs.image_process_result:
            return None

        image, yolo_results = obs.image_process_result
        if overlay:
            YoloClient.plot_results_ps(image, yolo_results)

        return image
    
    def execute_minispec(self, json_output: Stream | str):
        interpreter = MiniSpecInterpreter(self.message_queue, self.robot)
        interpreter.execute(json_output)

    def handle_task(self, user_instruction: str):
        # self._send_message('[TASK]: ' + user_instruction)
        self._send_message('Planning...')
        ret_val = None
        while True:
            t1 = time.time()
            self.current_plan = self.planner.plan(user_instruction)
            t2 = time.time()
            print_t(f"[C] Planning time: {t2 - t1:.2f}s")
            try:
                ret_val = self.execute_minispec(self.current_plan)
            except Exception as e:
                print_t(f"[C] Error: {e}")
            
            # disable replan for debugging
            break
            if ret_val.replan:
                print_t(f"[C] > Replanning <: {ret_val.value}")
                continue
            else:
                break
        
        self._send_message(f'\n[Task ended]')
        self._send_message('end')
        self.current_plan = None
        self.execution_history = None