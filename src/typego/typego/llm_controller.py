from PIL import Image
import queue, io, base64
from openai import Stream
from typing import Optional
import time
import threading
import random

from typego.llm_wrapper import ModelType
from typego.yolo_client import YoloClient
from typego.robot_observation import RobotPosture
from typego.plan_generator import PlanGenerator
from typego.utils import print_t
from typego.robot_info import RobotInfo
from typego.s2 import S2DPlan
from typego.s1 import S1
from typego.s0 import S0Event, S0
from typego.method import MethodEngine, make_find_object_method, make_follow_object_method

class LLMController():
    def __init__(self, robot_info: RobotInfo):
        self.running = False

        self.latest_inst = None
        self.latest_inst_lock = threading.Lock()

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
        self.s2d_loop_thread = threading.Thread(target=self.s2d_loop)

    def put_instruction(self, inst: str):
        with self.latest_inst_lock:
            print_t(f"[C] Received instruction: {inst}")
            self.latest_inst = inst
        
        # spawn a new thread to handle the instruction
        threading.Thread(target=self.on_new_task, args=(inst,), daemon=True).start()

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

        # self.s0_loop_thread.start()
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
    def on_new_task(self, inst: str):
        s2d_plan = S2DPlan(inst)

        # S1 plan starts
        s1_plan = S1(self.planner).plan(inst)
        print_t(f"[S1] Plan: {s1_plan}")
        s2d_plan.add_action(s1_plan)

        print(self.robot.registry.execute(s1_plan, callback=lambda r: s2d_plan.finish_action(r)))

        # Enter S2S loop
        loop_freq = 0.5
        count = 0
        while self.running and s2d_plan.is_active() and count < 5:
            count += 1
            print_t(f"[C] S2S loop for task {inst} ({s2d_plan.id}), current state: {s2d_plan.current_state}")
            start_time = time.time()
            plan = self.planner.s2s_plan(inst, s2d_plan, model_type=ModelType.GPT4O)

            print_t(f"[S2S] Plan: {plan}")
            next_action = s2d_plan.process_s2s_response(plan)
            print_t(f"[S2S] Action: {next_action}")
            if next_action:
                print(self.robot.registry.execute(next_action, callback=lambda r: s2d_plan.finish_action(r)))

            sleep_time = max(0, 1.0 / loop_freq - (time.time() - start_time))
            time.sleep(sleep_time)

        print_t(f"[C] Task {inst} ({s2d_plan.id}) completed or stopped.")

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