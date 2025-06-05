from PIL import Image
import queue, io, base64
from openai import Stream
from typing import Optional
import time
import threading

from typego.yolo_client import YoloClient
from typego.virtual_robot_wrapper import VirtualRobotWrapper
from typego.robot_wrapper import RobotPosture, RobotWrapper
from typego.llm_planner import LLMPlanner
from typego.utils import print_t, slam_map_overlay
from typego.minispec_interpreter import MiniSpecInterpreter
from typego.robot_info import RobotInfo

class S0Event:
    def __init__(self, description: str, condition: callable, action: callable, timeout: float = 3.0):
        self.description = description
        self.condition = condition
        self.action = action
        self.timeout = timeout
        self.timer = time.time()

    def check(self) -> bool:
        if self.condition() and (time.time() - self.timer) > 0.0:
            self.timer = time.time() + self.timeout
            return True
        return False
    
    def execute(self):
        self.action()

class LLMController():
    def __init__(self, robot_info: RobotInfo, message_queue: Optional[queue.Queue]=None):
        self.running = False
        self.message_queue = message_queue

        self.planner = LLMPlanner()

        self.controller_func = [
            self.user_log,
            self.probe
        ]

        self.s1_event = threading.Event()
        self.s2_event = threading.Event()

        self.s0_control = threading.Event()
        self.s0_control.set()

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

        self.s0_loop_thread = threading.Thread(target=self.s0_loop)
        self.s0_loop_thread.start()

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
    
    def s0_loop(self, rate: float = 20.0):
        delay = 1 / rate
        while not self.running:
            time.sleep(0.1)
        time.sleep(1.0)
        print_t(f"[C] Starting S0...")

        s0events = [
            # S0Event("Person found", lambda: self.robot.is_visible("person"), lambda: self.robot.look_up(), timeout=5),
            S0Event("Move back", lambda: self.robot.get_posture() == RobotPosture.STANDING and self.robot.observation.blocked(), lambda: self.robot.move(-0.3, 0.0), timeout=2.0),
        ]

        while self.running:
            for event in s0events:
                if event.check():
                    # block s1 and s2 loops
                    self.s0_control.clear()
                    print_t(f"[C] Get condition: {event.description}...")
                    # stop the current robot action
                    self.robot.stop_action()
                    event.execute()

                    self.s0_control.set()
            time.sleep(delay)

    def s1_loop(self, rate: float = 0.5):
        delay = 1 / rate
        while not self.running:
            time.sleep(0.1)
        time.sleep(1.0)
        print_t(f"[C] Starting S1...")
        while self.running:
            start_time = time.time()
            plan = self.planner.s1_plan()

            # pause if s0 is processing
            if not self.s0_control.is_set():
                print_t(f"[C] S0 is processing, waiting...")
                self.s0_control.wait()

            self.robot.append_actions(plan)
            print_t(f"[C] Plan: {plan}")
            
            elapsed = time.time() - start_time
            sleep_time = max(0, delay - elapsed)
            if sleep_time > 0:
                self.s1_event.wait(timeout=sleep_time)
            self.s1_event.clear()

    def s2_loop(self, rate: float = 0.2):
        delay = 1 / rate
        while not self.running:
            time.sleep(0.1)
        time.sleep(1.0)
        print_t(f"[C] Starting S2...")

        while self.running:
            start_time = time.time()
            plan = self.planner.s2_plan()
            print_t(f"[C] Plan: {plan}")

            # pause if s0 is processing
            if not self.s0_control.is_set():
                print_t(f"[C] S0 is processing, waiting...")
                self.s0_control.wait()

            self.robot.memory.process_s2_response(plan)

            elapsed = time.time() - start_time
            sleep_time = max(0, delay - elapsed)
            if sleep_time > 0:
                self.s2_event.wait(timeout=500.0)
            self.s2_event.clear()

    def user_instruction(self, inst: str):
        self.robot.memory.add_inst(inst)
        self.s1_event.set()  # Trigger the S1 loop to process the new instruction
        self.s2_event.set()  # Trigger the S2 loop to process the new instruction
