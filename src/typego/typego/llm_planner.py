import os, json
from typing import Optional

from typego.llm_wrapper import LLMWrapper, ModelType
from typego.utils import print_t
from typego.robot_wrapper import RobotWrapper
from typego.s2 import S2DPlan
from typego.yolo_client import ObservationEncoder

from ament_index_python.packages import get_package_share_directory
CURRENT_DIR = get_package_share_directory('typego')

CHAT_LOG_DIR = "/home/guojun/Documents/Go2-Livox-ROS2/src/typego/resource/"

def read_file(filename):
    try:
        filepath = os.path.join(CURRENT_DIR, "resource", filename)
        with open(filepath, "r", encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: File {filename} not found")
        return ""

class LLMPlanner():
    def __init__(self, robot: RobotWrapper):
        self.llm = LLMWrapper()
        self.robot = robot

        S2DPlan.init_default()

        self.s1_prompt = read_file("s1_prompt.txt")
        self.s1_examples = read_file("s1_examples.txt")

        self.s2s_prompt = read_file("s2s_prompt.txt")
        self.s2s_user_guidelines = read_file("s2s_user_guidelines.txt")
        self.s2s_examples = read_file("s2s_examples.txt")

        self.s2d_prompt = read_file("s2d_prompt.txt")
        self.s2d_user_guidelines = read_file("s2d_user_guidelines.txt")
        self.s2d_examples = read_file("s2d_examples.txt")

    def s1_plan(self, inst, model_type: ModelType = ModelType.LOCAL_1B) -> str:
        prompt = self.s1_prompt.format(example_plans=self.s1_examples,
                                        user_instruction=inst,
                                        observation=json.dumps(self.robot.observation.obs(), cls=ObservationEncoder, indent=2))

        ret = self.llm.request(prompt, model_type).split('\n')[0].strip()
        with open(CHAT_LOG_DIR + "s1_log.txt", "a") as f:
            remove_leading_prompt = prompt
            remove_leading_prompt += ret
            f.write(remove_leading_prompt + "\n---\n")
        return ret

    def s2s_plan(self, inst: Optional[str], model_type: ModelType = ModelType.GPT4O) -> str:
        robot_skills = "\n".join(self.robot.registry.get_skill_list())
        observation = json.dumps(self.robot.observation.obs(), cls=ObservationEncoder, indent=2)

        prompt = self.s2s_prompt.format(user_guidelines=self.s2s_user_guidelines,
                                            robot_skills=robot_skills,
                                            example_plans=self.s2s_examples,
                                            instruction=inst if inst else "None",
                                            current_plan=S2DPlan.CURRENT.get_s2s_input(),
                                            observation=observation)

        # print_t(f"[S2S] Execution request: {prompt.split('# CURRENT TASK', 1)[-1]}")
        try:
            ret = self.llm.request(prompt, model_type)
        except Exception as e:
            print_t(f"[S2S] Error during LLM request: {e}")
            return ""
        with open(CHAT_LOG_DIR + "s2s_log.txt", "a") as f:
            remove_leading_prompt = prompt.split("# CURRENT TASK", 1)[-1]
            remove_leading_prompt += ret
            f.write(remove_leading_prompt + "\n---\n")
        return ret

    def s2d_plan(self, inst: Optional[str], model_type: ModelType = ModelType.GPT4O) -> str:
        robot_skills = "\n".join(self.robot.registry.get_skill_list())
        observation = json.dumps(self.robot.observation.obs(), cls=ObservationEncoder, indent=2)

        prompt = self.s2d_prompt.format(user_guidelines=self.s2d_user_guidelines,
                                            example_plans=self.s2d_examples,
                                            user_instruction=inst,
                                            robot_skills=robot_skills,
                                            observation=observation)

        # print_t(f"[S2D] Execution request: {prompt.split('# CURRENT TASK', 1)[-1]}")
        ret = self.llm.request(prompt, model_type)
        with open(CHAT_LOG_DIR + "s2d_log.txt", "a") as f:
            remove_leading_prompt = prompt.split("# CURRENT TASK", 1)[-1]
            remove_leading_prompt += ret
            f.write(remove_leading_prompt + "\n---\n")
        return ret