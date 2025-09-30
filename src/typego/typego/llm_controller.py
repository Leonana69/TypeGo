from PIL import Image
import queue, io, base64
from openai import Stream
from typing import Optional
import time
import threading
import random

from typego.llm_wrapper import ModelType
from typego.yolo_client import YoloClient
from typego.robot_wrapper import RobotPosture
from typego.plan_generator import PlanGenerator
from typego.utils import print_t
from typego.robot_info import RobotInfo
from typego.s2 import S2DPlan
from typego.s0 import S0Event, S0
from typego.method import MethodEngine, make_find_object_method, make_follow_object_method

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

        if robot_info.robot_type == "virtual":
            from typego.virtual_robot_wrapper import VirtualRobotWrapper
            self.robot = VirtualRobotWrapper(robot_info)
        if robot_info.robot_type == "go2":
            from typego.go2_wrapper import Go2Wrapper
            self.robot = Go2Wrapper(robot_info)

        self.planner = PlanGenerator(self.robot)

        
        # self.vc_thread = threading.Thread(target=self.check_voice_command_thread)

        self.s0 = S0(self.robot)


        self.s0_loop_thread = threading.Thread(target=self.s0.loop)
        self.s1_loop_thread = threading.Thread(target=self.s1_loop)
        self.s2s_loop_thread = threading.Thread(target=self.s2s_loop)
        self.s2d_loop_thread = threading.Thread(target=self.s2d_loop)

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
        # self.s2s_loop_thread.start()
        # self.s2d_loop_thread.start()
        # self.vc_thread.start()

        # time.sleep(1.0)
        # self.robot.registry.execute("turn_right(180)")
        # self.robot.registry.execute("look_object('person')")
        # print(self.robot.registry.execute("move_forward(0.2)"))

    def stop_controller(self):
        self.running = False
        self.robot.stop()

    def fetch_robot_pov(self) -> Optional[Image.Image]:
        obs = self.robot.observation
        if not obs:
            print_t(f"[C] No observation available")
            return None

        process_result = obs.fetch_processed_result()
        if not process_result:
            print_t(f"[C] No processed result available")
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

            # time.sleep(3.0)
            # self.robot.registry.execute("stand_up()")

            print_t(f"[S1] Get plan: {plan}")

            self.robot.registry.execute('follow("sports ball")')
            # self.robot.registry.execute('turn_left(180)')

            # find_object_method = make_follow_object_method(self.robot)
            # find_person = find_object_method.bind(object="sports ball")
            # print(find_person.goal)
            # method_engine = MethodEngine(find_person)
            # result = method_engine.run()
            

            # print(f"[S1] Method result: {result}")
            # self.robot.registry.execute("nav(0.0, 0.0)")

            # self.robot.registry.execute("look_object(person)")

            # use S2DPlan.default to handle a new task
            # S2DPlan.set_default()
            # self.robot.append_actions(plan)

            ### TODO: spawn a new thread to do S2 planning

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
            plan = self.planner.s2s_plan(new_inst, model_type=ModelType.GROQ)

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
            t1 = time.time()
            plan = self.planner.s2d_plan(new_inst, model_type=ModelType.GROQ)
            print_t(f"[S2D] Plan: {plan}, took {time.time() - t1:.2f}s")

            # process plan ...
            S2DPlan.parse(new_inst, plan)
            print(S2DPlan.CURRENT)