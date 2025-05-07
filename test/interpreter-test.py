import sys
sys.path.append('..')
from typefly.platforms.virtual_robot_wrapper import VirtualRobotWrapper
from typefly.robot_wrapper import RobotWrapper
from typefly.llm_controller import LLMController
from typefly.minispec_interpreter import Statement, MiniSpecProgram
from typefly.robot_info import RobotInfo

# llm_output = '```json\n{\n    \"thoughts\": \"The user instruction is to turn around, which typically means a 180-degree rotation. This can be achieved by using the turn_cw skill with 180 degrees.\",\n    \"<plan, robot1>\": \"mf(180);tp()\"\n}\n```'
llm_output = "```json\n{\n    \"plan\": \"log('I cannot find an apple in the scene.')\"\n}\n```"

robot_info = RobotInfo("robot1", "virtual", {"capture": 0})
# controller = LLMController(robot_info, None)
robot = VirtualRobotWrapper(robot_info, [lambda x: (True, False), lambda x: (True, False)])

program = MiniSpecProgram(robot)
print(program.parse(llm_output, True))
print(program.eval())