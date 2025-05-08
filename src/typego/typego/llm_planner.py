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

        with open(os.path.join(CURRENT_DIR, f"./resource/s1_plan_prompt.txt"), "r") as f:
            self.s1_plan_prompt = f.read()

        with open(os.path.join(CURRENT_DIR, f"./resource/s1_code_prompt.txt"), "r") as f:
            self.s1_code_prompt = f.read()

        with open(os.path.join(CURRENT_DIR, f"./resource/s1_guidelines.txt"), "r") as f:
            self.s1_guidelines = f.read()

        with open(os.path.join(CURRENT_DIR, f"./resource/s1_plan_examples.txt"), "r") as f:
            self.s1_plan_examples = f.read()

        with open(os.path.join(CURRENT_DIR, f"./resource/s1_code_examples.txt"), "r") as f:
            self.s1_code_examples = f.read()

        with open(os.path.join(CURRENT_DIR, f"./resource/prompt_probe.txt"), "r") as f:
            self.prompt_probe = f.read()

    def set_robot(self, robot: RobotWrapper):
        self.robot = robot

    def plan_subtask(self, user_instruction: str):
        scene_description = self.robot.get_obj_list_str() + "\n"
        
        prompt = self.s1_plan_prompt.format(example_plans=self.s1_plan_examples,
                                         user_instruction=user_instruction,
                                         scene_description=scene_description)

        return self.llm.request(prompt, self.model_type, stream=False)

    def plan(self, user_instruction: str):
        robot_skills = ""
        scene_description = self.robot.get_obj_list_str() + "\n"
        exec_history_str = ""

        robot_skills += f"#### Low-level skills\n"
        robot_skills += str(self.robot.ll_skillset)
        if self.robot.hl_skillset is not None:
            robot_skills += f"\n#### High-level skills\n"
            robot_skills += str(self.robot.hl_skillset)

        exec_history = self.robot.memory.get()

        print_t(f"[P] Execution history: {exec_history}")

        if len(exec_history) == 0:
            exec_history_str += "None\n"
        else:
            exec_history_str += "Plan | Result\n"
            exec_history_str += "-----|-------\n"
            for action in exec_history:
                exec_history_str += f"{action}\n"

        print(f"[P] Execution history: {exec_history_str}")
        prompt = self.s1_code_prompt.format(guidelines=self.s1_guidelines,
                                         robot_skills=robot_skills,
                                         example_plans=self.s1_code_examples,
                                         user_instruction=user_instruction,
                                         current_subtask="",
                                         execution_history=exec_history_str,
                                         scene_description=scene_description)

        return self.llm.request(prompt, self.model_type, stream=False)
    
    def probe(self, query: str) -> str:
        prompt = self.prompt_probe.format(scene_description=self.robot.get_obj_list_str(), query=query)
        print_t(f"[P] Execution request: {query}")
        return self.llm.request(prompt, self.model_type)