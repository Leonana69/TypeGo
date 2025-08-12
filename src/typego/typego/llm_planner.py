import os, json

from typego.llm_wrapper import LLMWrapper, ModelType
from typego.utils import print_t
from typego.robot_wrapper import RobotWrapper
from typego.s2 import S2Plan

from ament_index_python.packages import get_package_share_directory
CURRENT_DIR = get_package_share_directory('typego')

CHAT_LOG_DIR = "/home/guojun/Documents/Go2-Livox-ROS2/src/typego/resource/"

class LLMPlanner():
    def __init__(self, model_type: ModelType = ModelType.GPT4O):
        self.llm = LLMWrapper()
        self.model_type = model_type

        S2Plan.init_default()

        self.s0_prompt = open(os.path.join(CURRENT_DIR, f"./resource/s0_prompt.txt"), "r").read()
        self.s0_examples = open(os.path.join(CURRENT_DIR, f"./resource/s0_examples.txt"), "r").read()

        self.s1_prompt = open(os.path.join(CURRENT_DIR, f"./resource/s1_prompt.txt"), "r").read()
        self.s1_user_guidelines = open(os.path.join(CURRENT_DIR, f"./resource/s1_user_guidelines.txt"), "r").read()
        self.s1_examples = open(os.path.join(CURRENT_DIR, f"./resource/s1_examples.txt"), "r").read()

        self.s2_prompt = open(os.path.join(CURRENT_DIR, f"./resource/s2_prompt.txt"), "r").read()
        self.s2_user_guidelines = open(os.path.join(CURRENT_DIR, f"./resource/s2_user_guidelines.txt"), "r").read()
        self.s2_examples = open(os.path.join(CURRENT_DIR, f"./resource/s2_examples.txt"), "r").read()
    
    def set_robot(self, robot: RobotWrapper):
        self.robot = robot

    def s0_plan(self, inst) -> str:
        state = "State: " + self.robot.get_state()

        scene_description = "Objects: " + self.robot.get_obj_list_str() + "\n\n"
        scene_description += "Waypoints: " + self.robot.observation.slam_map.get_waypoint_list_str()

        prompt = self.s0_prompt.format(example_plans=self.s0_examples,
                                            user_instruction=inst,
                                            # robot_skills=robot_skills,
                                            robot_state=state,
                                            scene_description=scene_description)

        ret = self.llm.request(prompt, ModelType.LOCAL_1B)
        with open(CHAT_LOG_DIR + "s0_log.txt", "a") as f:
            remove_leading_prompt = prompt#.split("# CURRENT TASK", 1)[-1]
            remove_leading_prompt += ret
            f.write(remove_leading_prompt + "\n---\n")
        return ret

    def s1_plan(self, inst: str | None) -> str:
        robot_skills = ""
        robot_skills += f"#### Skills\n"
        robot_skills += "\n".join(self.robot.registry.get_skill_list())

        state = "State: " + self.robot.get_state() + "\n\n"
        print_t(f"[S1] Current S2: {S2Plan.CURRENT.content}")
        state += "Plan for Current State: " + S2Plan.CURRENT.get_s1_input() + "\n\n"
        state += "Action History for Current Instruction: " + str(S2Plan.CURRENT.action_list) + "\n\n"

        scene_description = "Objects: " + self.robot.get_obj_list_str() + "\n\n"
        scene_description += "Waypoints: " + self.robot.observation.slam_map.get_waypoint_list_str()

        prompt = self.s1_prompt.format(user_guidelines=self.s1_user_guidelines,
                                            robot_skills=robot_skills,
                                            example_plans=self.s1_examples,
                                            instruction=inst if inst else "None",
                                            robot_state=state,
                                            scene_description=scene_description)

        # print_t(f"[S1] Execution request: {prompt.split('# CURRENT TASK', 1)[-1]}")
        try:
            ret = self.llm.request(prompt, self.model_type)
        except Exception as e:
            print_t(f"[S1] Error during LLM request: {e}")
            return ""
        with open(CHAT_LOG_DIR + "s1_log.txt", "a") as f:
            remove_leading_prompt = prompt.split("# CURRENT TASK", 1)[-1]
            remove_leading_prompt += ret
            f.write(remove_leading_prompt + "\n---\n")
        return ret

    def s2_plan(self, inst) -> str:
        robot_skills = ""
        robot_skills += f"#### Low-level skills\n"
        robot_skills += str(self.robot.ll_skillset._sim())
        if self.robot.hl_skillset is not None:
            robot_skills += f"\n#### High-level skills\n"
            robot_skills += str(self.robot.hl_skillset._sim())

        state = "State: " + self.robot.get_state() + "\n\n"
        state += "Instruction History: [\n" + ", ".join(str(p) for p in S2Plan.get_history_sorted_by_time()) + "\n]\n"

        scene_description = "Objects: " + self.robot.get_obj_list_str() + "\n\n"
        scene_description += "Waypoints: " + self.robot.observation.slam_map.get_waypoint_list_str()

        prompt = self.s2_prompt.format(user_guidelines=self.s2_user_guidelines,
                                            example_plans=self.s2_examples,
                                            user_instruction=inst,
                                            robot_skills=robot_skills,
                                            robot_state=state,
                                            scene_description=scene_description)

        # print_t(f"[S2] Execution request: {prompt.split('# CURRENT TASK', 1)[-1]}")
        ret = self.llm.request(prompt, self.model_type, image=self.robot.observation.slam_map.get_map())
        with open(CHAT_LOG_DIR + "s2_log.txt", "a") as f:
            remove_leading_prompt = prompt.split("# CURRENT TASK", 1)[-1]
            remove_leading_prompt += ret
            f.write(remove_leading_prompt + "\n---\n")
        return ret