from PIL import Image
import queue, io, base64
from openai import Stream
from typing import Optional
import time
import threading

from typego.yolo_client import YoloClient
from typego.robot_wrapper import RobotPosture
from typego.llm_planner import LLMPlanner
from typego.utils import print_t
from typego.robot_info import RobotInfo
from typego.s2 import S2DPlan
from typego.method import MethodEngine, make_find_object_method, make_follow_object_method

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
    def __init__(self, robot_info: RobotInfo):
        self.running = False

        self.latest_inst = None
        self.latest_inst_lock = threading.Lock()
        self.latest_inst_flag = [False, False, False]  # [s0, s1, s2]

        self.s0_event = threading.Event()
        self.s1_s2s_event = threading.Event()
        self.s1_s2s_event.set()
        self.s1_event = threading.Event()
        self.s2s_event = threading.Event()
        self.s2d_event = threading.Event()

        self.s0_control = threading.Event()
        self.s0_control.set()

        if robot_info.robot_type == "virtual":
            from typego.virtual_robot_wrapper import VirtualRobotWrapper
            self.robot = VirtualRobotWrapper(robot_info)
        if robot_info.robot_type == "go2":
            from typego.go2_wrapper import Go2Wrapper
            self.robot = Go2Wrapper(robot_info)
        
        self.planner = LLMPlanner(self.robot)

        self.s0_loop_thread = threading.Thread(target=self.s0_loop)
        self.s1_loop_thread = threading.Thread(target=self.s1_loop)
        self.s2s_loop_thread = threading.Thread(target=self.s2s_loop)
        self.s2d_loop_thread = threading.Thread(target=self.s2d_loop)
        # self.vc_thread = threading.Thread(target=self.check_voice_command_thread)

    def check_voice_command_thread(self):
        while not self.running:
            time.sleep(0.1)
        time.sleep(1.0)
        while self.running:
            command = self.robot.observation.fetch_command()  # Non-blocking fetch
            if command:
                print_t(f"[C] Received voice command: {command}")
                self.put_instruction(command)
            time.sleep(0.1)

    def start_controller(self):
        self.running = True
        self.robot.start()

        self.s0_loop_thread.start()
        self.s1_loop_thread.start()
        self.s2s_loop_thread.start()
        self.s2d_loop_thread.start()
        # self.vc_thread.start()

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
        print_t(f"[C] Starting S0...")

        s0events = [
            # S0Event("Person found", lambda: self.robot.is_visible("person"), lambda: self.robot.look_up(), 5, timeout=10),
            S0Event("Look sports ball", lambda: self.robot.is_visible("sports ball"), lambda: self.robot.look_object('sports ball'), 1, timeout=3),
            S0Event("Step back", lambda: self.robot.observation.blocked(), lambda: self.robot.move(-0.3, 0.0), 0, timeout=1.0),
        ]

        self.s0_in_progress_event: Optional[S0Event] = None
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

    """
    S1: Fast thinking, react to new observations, adjust short-term plans
    """
    def s1_loop(self):
        print_t(f"[C] Starting S1...")
        while self.running:
            # Current S1 is triggered by new instruction
            self.s1_event.wait()
            print_t(f"[S1] Received S1 event, processing...")
            self.s1_event.clear()
            # block S2S for a moment
            self.s1_s2s_event.clear()
            new_inst = self.get_instruction(0)
            plan = self.planner.s1_plan(new_inst)
            print_t(f"[S1] Get plan: {plan}")

            # find_object_method = make_follow_object_method(self.robot)
            # find_person = find_object_method.bind(object="sports ball")
            # print(find_person.goal)
            # method_engine = MethodEngine(find_person)
            # result = method_engine.run()
            # print(f"[S1] Method result: {result}")
            # self.robot.registry.execute("nav(0.0, 0.0)")

            # use S2DPlan.default to handle a new task
            # S2DPlan.set_default()
            # self.robot.append_actions(plan)

            # allow S2S to proceed
            self.s1_s2s_event.set()

    def s2s_loop(self, rate: float = 0.5):
        delay = 1 / rate
        print_t(f"[C] Starting S2S...")

        while self.running:
            start_time = time.time()

            new_inst = self.get_instruction(1)
            if new_inst is None:
                self.s1_s2s_event.wait(timeout=100)
            plan = self.planner.s2s_plan(new_inst)

            # pause if s0 is processing
            # if not self.s0_control.is_set():
            #     print_t(f"[C] S0 is processing, waiting...")
            #     self.s0_control.wait()

            # self.robot.append_actions(plan)
            print_t(f"[S2S] Plan: {plan}")
            action = S2DPlan.CURRENT.process_s1_response(plan)
            print_t(f"[S2S] Action: {action}")
            if action:
                self.robot.append_actions(action)
            
            sleep_time = max(0, delay - (time.time() - start_time))
            self.s1_event.wait(timeout=sleep_time)
            self.s1_event.clear()

    def s2d_loop(self, rate: float = 0.2):
        delay = 1 / rate
        print_t(f"[C] Starting S2D...")

        while self.running:
            self.s2d_event.wait()
            self.s2d_event.clear()

            new_inst = self.get_instruction(2)
            plan = self.planner.s2d_plan(new_inst)
            print_t(f"[S2D] Plan: {plan}")

            # process plan ...
            S2DPlan.parse(new_inst, plan)
            print(S2DPlan.CURRENT)

    def get_instruction(self, index: int) -> Optional[str]:
        with self.latest_inst_lock:
            if not self.latest_inst_flag[index]:
                self.latest_inst_flag[index] = True
                return self.latest_inst
            else:
                return None

    def put_instruction(self, inst: str):
        with self.latest_inst_lock:
            print_t(f"[C] Received instruction: {inst}")
            self.latest_inst = inst
            self.latest_inst_flag = [False, False, False]  # [s1, s2s, s2d]
            self.s1_event.set()
            self.s2s_event.set()
            self.s2d_event.set()