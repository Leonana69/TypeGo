import os

from typego.llm_wrapper import LLMWrapper, ModelType
from typego.utils import print_t
from typego.robot_wrapper import RobotWrapper

from ament_index_python.packages import get_package_share_directory
CURRENT_DIR = get_package_share_directory('typego')

CHAT_LOG_DIR = "/home/guojun/Documents/Go2-Livox-ROS2/src/typego/resource/"

class LLMPlanner():
    def __init__(self, model_type: ModelType = ModelType.GPT4O):
        self.llm = LLMWrapper()
        self.model_type = model_type

        with open(os.path.join(CURRENT_DIR, f"./resource/prompt_probe.txt"), "r") as f:
            self.prompt_probe = f.read()

        self.s1_prompt = open(os.path.join(CURRENT_DIR, f"./resource/s1_prompt.txt"), "r").read()
        self.s1_user_guidelines = open(os.path.join(CURRENT_DIR, f"./resource/s1_user_guidelines.txt"), "r").read()
        self.s1_examples = open(os.path.join(CURRENT_DIR, f"./resource/s1_examples.txt"), "r").read()

        self.s2_prompt = open(os.path.join(CURRENT_DIR, f"./resource/s2_prompt.txt"), "r").read()
        self.s2_user_guidelines = open(os.path.join(CURRENT_DIR, f"./resource/s2_user_guidelines.txt"), "r").read()
        self.s2_examples = open(os.path.join(CURRENT_DIR, f"./resource/s2_examples.txt"), "r").read()
    
    def set_robot(self, robot: RobotWrapper):
        self.robot = robot

    def s1_plan(self):
        robot_skills = ""

        robot_skills += f"#### Low-level skills\n"
        robot_skills += str(self.robot.ll_skillset)
        if self.robot.hl_skillset is not None:
            robot_skills += f"\n#### High-level skills\n"
            robot_skills += str(self.robot.hl_skillset)

        state = "State: " + self.robot.get_state() + "\n\n"
        state += "Plan for Current Instruction: " + self.robot.memory.get_current_plan_str() + "\n\n"
        state += "Action History for Current Instruction: " + self.robot.memory.get_history_action_str() + "\n\n"

        inst = self.robot.memory.get_current_inst_str()

        prompt = self.s1_prompt.format(user_guidelines=self.s1_user_guidelines,
                                            robot_skills=robot_skills,
                                            example_plans=self.s1_examples,
                                            instruction=inst,
                                            robot_state=state,
                                            scene_description=self.robot.get_obj_list_str() + "\n")

        # print_t(f"[S1] Execution request: {prompt.split('# CURRENT TASK', 1)[-1]}")
        ret = self.llm.request(prompt, self.model_type, stream=False)
        with open(CHAT_LOG_DIR + "s1_log.txt", "a") as f:
            remove_leading_prompt = prompt.split("# CURRENT TASK", 1)[-1]
            remove_leading_prompt += ret
            f.write(remove_leading_prompt + "\n---\n")
        return ret

    def s2_plan(self) -> str:
        inst = self.robot.memory.get_current_inst_str()

        scene_description = "Objects: " + self.robot.get_obj_list_str() + "\n\n"
        scene_description += "Waypoints: " + self.robot.observation.slam_map.get_waypoint_list_str()

        state = "State: " + self.robot.get_state() + "\n\n"
        state += "Instruction History: " + self.robot.memory.get_inst_list_str() + "\n\n"
        state += "Plan for In-Progress Instruction: " + self.robot.memory.get_current_plan_str() + "\n\n"
        state += "Action History for In-Progress Instruction: " + self.robot.memory.get_history_action_str() + "\n\n"

        prompt = self.s2_prompt.format(user_guidelines=self.s2_user_guidelines,
                                            example_plans=self.s2_examples,
                                            user_instruction=inst,
                                            robot_state=state,
                                            scene_description=scene_description)

        # print_t(f"[S2] Execution request: {prompt.split('# CURRENT TASK', 1)[-1]}")
        ret = self.llm.request(prompt, self.model_type, stream=False)
        with open(CHAT_LOG_DIR + "s2_log.txt", "a") as f:
            remove_leading_prompt = prompt.split("# CURRENT TASK", 1)[-1]
            remove_leading_prompt += ret
            f.write(remove_leading_prompt + "\n---\n")
        return ret

    def probe(self, query: str) -> str:
        prompt = self.prompt_probe.format(scene_description=self.robot.get_obj_list_str(), query=query)
        print_t(f"[P] Execution request: {query}")
        return self.llm.request(prompt, self.model_type)