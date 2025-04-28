import sys
sys.path.append('..')
from typefly.platforms.virtual_robot_wrapper import VirtualRobotWrapper
from typefly.robot_wrapper import RobotWrapper
from typefly.llm_controller import LLMController
from typefly.minispec_interpreter import Statement
from typefly.robot_info import RobotInfo

# llm_output = '```json\n{\n    \"thoughts\": \"The user instruction is to turn around, which typically means a 180-degree rotation. This can be achieved by using the turn_cw skill with 180 degrees.\",\n    \"<plan, robot1>\": \"mf(180);tp()\"\n}\n```'
llm_output = '{$is_visible("person") {rotate(180)}$is_visible("bottle") {move(90,180)}move(180, 0);take_picture()}'

controller = LLMController([], None)
robot = VirtualRobotWrapper(RobotInfo("robot1", "virtual", {"capture": 0}), [lambda x: (True, False), lambda x: (True, False)])

program = Statement(None, robot)
print(program.parse(llm_output, True))
print(program.eval())
print(program.to_string())