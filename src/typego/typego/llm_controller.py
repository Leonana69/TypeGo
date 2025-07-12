from PIL import Image
import queue, io, base64
from openai import Stream
from typing import Optional
import time
import threading

from typego.yolo_client import YoloClient
from typego.virtual_robot_wrapper import VirtualRobotWrapper
from typego.robot_wrapper import RobotPosture
from typego.llm_planner import LLMPlanner
from typego.utils import print_t
from typego.robot_info import RobotInfo

class S0Event:
    def __init__(self, description: str, condition: callable, action: callable, priority: int = 0, timeout: float = 3.0):
        self.description = description
        self.condition = condition
        self.action = action
        self.timeout = timeout
        self.priority = priority
        self.timer = time.time()

    def check(self, current_priority: int=99) -> bool:
        if self.priority < current_priority and self.condition() and (time.time() - self.timer) > 0.0:
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

        # self.s2_loop_thread = threading.Thread(target=self.s2_loop)
        # self.s2_loop_thread.start()

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
        if not obs:
            return None

        process_result = obs.fetch_processed_result()
        if not process_result:
            return None

        image, yolo_results = process_result
        image = YoloClient.plot_results_ps(image.copy(), yolo_results)
        return image
    
    def fetch_robot_map(self) -> Optional[Image.Image]:
        obs = self.robot.observation
        if not obs or obs.slam_map.is_empty():
            return None

        image = obs.slam_map.get_map()
        return Image.fromarray(image)
    
    def s0_loop(self, rate: float = 100.0):
        delay = 1 / rate
        while not self.running:
            time.sleep(0.1)
        time.sleep(1.0)
        print_t(f"[C] Starting S0...")

        s0events = [
            # S0Event("Person found", lambda: self.robot.is_visible("person"), lambda: self.robot.look_up(), 5, timeout=10),
            S0Event("Look sports ball", lambda: self.robot.is_visible("sports ball"), lambda: self.robot.look_object('sports ball'), 1, timeout=3),
            S0Event("Step back", lambda: self.robot.observation.blocked(), lambda: self.robot.move(-0.3, 0.0), 0, timeout=1.0),
        ]

        self.s0_in_progress_event: S0Event | None = None
        event_queue = queue.Queue()
        s0_event_executor_thread = threading.Thread(target=self.s0_event_executor, args=(event_queue,))
        s0_event_executor_thread.start()

        while self.running:
            for event in s0events:
                if event.check(self.s0_in_progress_event.priority if self.s0_in_progress_event else 99):
                    if self.s0_in_progress_event is None:
                        print_t(f"[C] New event triggered: {event.description}")
                    else:
                        print_t(f"[C] Canceling {self.s0_in_progress_event.description}... and executing {event.description}")
                        self.robot.stop_action()

                    event_queue.put(event)
            time.sleep(delay)

    def s0_event_executor(self, event_queue: queue.Queue[S0Event]):
        while self.running:
            try:
                event = event_queue.get(timeout=1)
                self.s0_control.clear()
                self.s0_in_progress_event = event
                event.execute()
                self.s0_in_progress_event = None
                self.s0_control.set()
            except queue.Empty:
                continue

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
            print_t(f"[S1] Plan: {plan}")
            
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
            print_t(f"[S2] Plan: {plan}")

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
