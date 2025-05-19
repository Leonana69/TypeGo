import os

from typego.llm_wrapper import LLMWrapper, ModelType
from typego.utils import print_t
from typego.robot_wrapper import RobotWrapper

from ament_index_python.packages import get_package_share_directory
CURRENT_DIR = get_package_share_directory('typego')

class LLMPlanner():
    def __init__(self, model_type: ModelType = ModelType.GPT4O):
        self.llm = LLMWrapper()
        self.model_type = model_type

        # with open(os.path.join(CURRENT_DIR, f"./resource/s1_plan_prompt.txt"), "r") as f:
        #     self.s1_plan_prompt = f.read()

        # with open(os.path.join(CURRENT_DIR, f"./resource/s1_code_prompt.txt"), "r") as f:
        #     self.s1_code_prompt = f.read()

        # with open(os.path.join(CURRENT_DIR, f"./resource/s1_guidelines.txt"), "r") as f:
        #     self.s1_guidelines = f.read()

        # with open(os.path.join(CURRENT_DIR, f"./resource/s1_plan_examples.txt"), "r") as f:
        #     self.s1_plan_examples = f.read()

        # with open(os.path.join(CURRENT_DIR, f"./resource/s1_code_examples.txt"), "r") as f:
        #     self.s1_code_examples = f.read()

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

    def s1_plan(self, instruction: str):
        robot_skills = ""

        robot_skills += f"#### Low-level skills\n"
        robot_skills += str(self.robot.ll_skillset)
        if self.robot.hl_skillset is not None:
            robot_skills += f"\n#### High-level skills\n"
            robot_skills += str(self.robot.hl_skillset)

        prompt = self.s1_prompt.format(user_guidelines=self.s1_user_guidelines,
                                            robot_skills=robot_skills,
                                            example_plans=self.s1_examples,
                                            s1_instruction=instruction,
                                            robot_state=self.robot.get_state(),
                                            scene_description=self.robot.get_obj_list_str() + "\n")

        return self.llm.request(prompt, self.model_type, stream=False)
    
    def s2_plan(self, instruction: str):
        scene_description = self.robot.get_obj_list_str() + "\n"
        scene_description += self.robot.observation.slam_map.get_waypoint_list_str()

        prompt = self.s2_prompt.format(user_guidelines=self.s2_user_guidelines,
                                            example_plans=self.s2_examples,
                                            user_instruction=instruction,
                                            robot_state=self.robot.get_state(),
                                            scene_description=scene_description)

        print_t(f"[P] Execution request: {prompt}")
        # return self.llm.request(prompt, self.model_type, stream=False)
        return "keep()"
    
    def probe(self, query: str) -> str:
        prompt = self.prompt_probe.format(scene_description=self.robot.get_obj_list_str(), query=query)
        print_t(f"[P] Execution request: {query}")
        return self.llm.request(prompt, self.model_type)