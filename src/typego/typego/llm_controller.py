from PIL import Image
import queue, io, base64
from openai import Stream
from typing import Optional
import time
import threading
import random, json

from typego.llm_wrapper import ModelType
from typego.yolo_client import YoloClient
from typego.robot_observation import RobotPosture, ObservationEncoder
from typego.plan_generator import PlanGenerator
from typego.utils import print_t
from typego.robot_info import RobotInfo
from typego.s2 import S2DPlan
from typego.s1 import S1
from typego.s0 import S0Event, S0
from typego.method import MethodEngine, make_find_object_method, make_follow_object_method
import typego.frontend_message as frontend_message

class LLMController():
    def __init__(self, robot_info: RobotInfo):
        self.running = False

        self.s2d_event = threading.Event()

        if robot_info.robot_type == "virtual":
            from typego.virtual_robot import VirtualRobot
            self.robot = VirtualRobot(robot_info)
        if robot_info.robot_type == "go2":
            from typego.go2_wrapper import Go2
            self.robot = Go2(robot_info)

        self.planner = PlanGenerator(self.robot)
        
        # self.vc_thread = threading.Thread(target=self.check_voice_command_thread)

        self.s0 = S0(self.robot)
        self.s0_loop_thread = threading.Thread(target=self.s0.loop)
        self.s2d_loop_thread = threading.Thread(target=self.s2d_loop)

    def put_instruction(self, inst: str) -> int:
        print_t(f"[C] Received instruction: {inst}")
        
        # spawn a new thread to handle the instruction
        s2d_plan = S2DPlan(inst)
        threading.Thread(target=self.on_new_task, args=(s2d_plan,), daemon=True).start()
        return s2d_plan.id

    def check_voice_command_thread(self):
        while not self.running:
            time.sleep(0.1)
        time.sleep(1.0)
        while self.running:
            command = self.robot.obs.fetch_command()  # Non-blocking fetch
            if command:
                print_t(f"[C] Received voice command: {command}")
                self.put_instruction(command)
            time.sleep(0.1)

    def start_controller(self):
        self.running = True
        self.robot.start()

        # self.s0_loop_thread.start()
        self.s2d_loop_thread.start()
        # self.vc_thread.start()

        # time.sleep(1.0)
        # self.robot.registry.execute("turn_right(180)", task_id=123)
        # time.sleep(1.0)
        # self.robot.registry.execute("look_object('person')", task_id=123)
        # print(self.robot.registry.execute("move_forward(0.6)", task_id=124))

        time.sleep(1.0)
        test_plan = S2DPlan.init_test_plan()
        threading.Thread(target=self.on_new_task, args=(test_plan,), daemon=True).start()

    def stop_controller(self):
        self.running = False
        self.robot.stop()

    def fetch_robot_pov(self) -> Optional[Image.Image]:
        obs = self.robot.obs
        if not obs:
            print_t(f"[C] No observation available")
            return None

        process_result = obs.fetch_objects()
        if not process_result:
            # print_t(f"[C] No processed result available")
            return None

        image, yolo_results = process_result
        image = YoloClient.plot_results_ps(image.copy(), yolo_results)
        return image
    
    def fetch_robot_map(self) -> Optional[Image.Image]:
        obs = self.robot.obs
        if not obs or obs.slam_map.is_empty():
            return None

        image = obs.slam_map.get_map()
        return Image.fromarray(image)

    def on_new_task(self, s2d_plan: S2DPlan):
        frontend_message.publish(f"Received new task: {s2d_plan.content}", task_id=s2d_plan.id)
        inst = s2d_plan.content
        # S1 plan starts
        s1_plan = S1(self.planner).plan(inst)
        print_t(f"[S2S ({s2d_plan.id})] S1 plan: {s1_plan}")
        s2d_plan.add_action(s1_plan)

        # Notify S2D loop
        self.s2d_event.set()

        s1_exec_ret = self.robot.registry.execute(s1_plan, task_id=s2d_plan.id, callback=lambda r: s2d_plan.finish_action(r))
        print_t(f'[S2S ({s2d_plan.id})] S1 execution ret: {s1_exec_ret}')

        # Enter S2S loop
        self.s2s_loop(s2d_plan)

    def s2s_loop(self, s2d_plan: S2DPlan, rate: float = 1.0):
        plan_count = 0
        while self.running and s2d_plan.is_active():
            plan_count += 1
            while not s2d_plan.is_running() and self.running:
                print_t(f"[S2S ({s2d_plan.id})] Task paused. Waiting to resume...")
                time.sleep(1.0)

            print_t(f"[S2S ({s2d_plan.id})] Loop for task {s2d_plan.content}, current state: {s2d_plan.current_state}")
            start_time = time.time()
            plan = self.planner.s2s_plan(s2d_plan.content, s2d_plan, model_type=ModelType.GROQ)

            print_t(f"[S2S ({s2d_plan.id})] Plan: {plan}")
            next_action = s2d_plan.process_s2s_response(plan)
            print_t(f"[S2S ({s2d_plan.id})] Action: {next_action}")
            if next_action:
                exec_ret = self.robot.registry.execute(next_action, task_id=s2d_plan.id, callback=lambda r: s2d_plan.finish_action(r))
                print_t(f'[S2S ({s2d_plan.id})] Execution ret: {exec_ret}')

            sleep_time = max(0, 1.0 / rate - (time.time() - start_time))
            time.sleep(sleep_time)

        print_t(f"[S2S ({s2d_plan.id})] Task completed or stopped.")
        frontend_message.end_queue(s2d_plan.id)

    def s2d_loop(self, rate: float = 0.5):
        delay = 1 / rate
        print_t(f"[S2D] Starting S2D...")

        planning_time = 0
        while self.running:
            self.s2d_event.wait(timeout=max(0, delay - planning_time))
            self.s2d_event.clear()

            print_t(f"[S2D] Loop triggered. {S2DPlan.get_s2d_input()}")

            start_time = time.time()
            plan = self.planner.s2d_plan(model_type=ModelType.GROQ)

            optional_new_plan = S2DPlan.process_s2d_response(plan)
            if optional_new_plan is not None:
                print_t(f"[S2D] New S2D plan detected: {optional_new_plan.content}")
                threading.Thread(target=self.s2s_loop, args=(optional_new_plan,), daemon=True).start()

            planning_time += time.time() - start_time