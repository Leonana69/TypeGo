from PIL import Image
import queue, io, base64
from openai import Stream
from typing import Optional
import time
import threading

from typego.yolo_client import YoloClient
from typego.virtual_robot_wrapper import VirtualRobotWrapper
from typego.robot_wrapper import RobotWrapper
from typego.llm_planner import LLMPlanner
from typego.utils import print_t, slam_map_overlay
from typego.minispec_interpreter import MiniSpecInterpreter
from typego.robot_info import RobotInfo

class LLMController():
    def __init__(self, robot_info: RobotInfo, message_queue: Optional[queue.Queue]=None):
        self.running = False
        self.message_queue = message_queue

        self.planner = LLMPlanner()

        self.controller_func = [
            self.user_log,
            self.probe
        ]

        self.inst_queue = queue.Queue()
        self.subtask_queue = queue.Queue()

        if robot_info.robot_type == "virtual":
            self.robot = VirtualRobotWrapper(robot_info, self.controller_func)
        elif robot_info.robot_type == "go2":
            from typego.go2_wrapper import Go2Wrapper
            self.robot = Go2Wrapper(robot_info, self.controller_func)
        
        self.planner.set_robot(self.robot)

        self.s1_loop_thread = threading.Thread(target=self.s1_loop)
        self.s1_loop_thread.start()

        self.s2_loop_thread = threading.Thread(target=self.s2_loop)
        self.s2_loop_thread.start()

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
        self.running = True
        self.robot.start()
        
    def stop_controller(self):
        self.running = False
        self.robot.stop()

    def fetch_robot_pov(self) -> Optional[Image.Image]:
        obs = self.robot.observation
        if not obs or not obs.image_process_result:
            return None

        image, yolo_results = obs.image_process_result
        image = YoloClient.plot_results_ps(image.copy(), yolo_results)

        return image
    
    def fetch_robot_map(self) -> Optional[Image.Image]:
        obs = self.robot.observation
        if not obs or obs.slam_map.is_empty():
            return None

        image = obs.slam_map.get_map()
        return Image.fromarray(image)
    
    def s1_loop(self, rate: float = 0.2):
        delay = 1 / rate
        while not self.running:
            time.sleep(0.1)
        time.sleep(1.0)
        print_t(f"[C] Starting continuous planning...")
        while self.running:
            start_time = time.time()
            plan = self.planner.s1_plan()
            self.robot.append_action(plan)
            print_t(f"[C] Plan: {plan}")
            
            elapsed = time.time() - start_time
            sleep_time = max(0, delay - elapsed)
            time.sleep(sleep_time)

    def s2_loop(self, rate: float = 0.1):
        delay = 1 / rate
        while not self.running:
            time.sleep(0.1)
        time.sleep(1.0)
        print_t(f"[C] Starting continuous planning...")

        while self.running:
            start_time = time.time()
            new_inst = None
            try:
                new_inst = self.inst_queue.get_nowait()
                self.robot.memory.new_instruction(new_inst)
            except queue.Empty:
                pass

            plan = self.planner.s2_plan(new_inst)
            print_t(f"[C] Plan: {plan}")
            self.robot.memory.process_s2_response(plan)
            elapsed = time.time() - start_time
            sleep_time = max(0, delay - elapsed)
            time.sleep(sleep_time)

    def user_instruction(self, inst: str):
        self.inst_queue.put(inst)