from dataclasses import dataclass
import re, queue
import copy
from enum import Enum, auto
import time
from threading import Thread, Event
from openai import Stream

from typego.robot_wrapper import RobotWrapper
from typego.robot_info import RobotInfo
from typego.skill_item import SKILL_RET_TYPE
from typego.utils import print_t, evaluate_value

def _print_debug(*args):
    print(*args)
    # pass

EXECUTION_QUEUE: queue.Queue = queue.Queue()

class ProgramParsingState(Enum):
    NONE = auto()
    JSON_BEGIN = auto()
    PREFIX = auto()
    ROBOT_ID = auto()
    QUOTATION_START = auto()
    PLAN = auto()
    QUOTATION_END = auto()
    JSON_END = auto()

class MiniSpecProgram:
    def __init__(self, robot: RobotWrapper, message_queue: queue.Queue=None) -> None:
        self.parse_state: ProgramParsingState = ProgramParsingState.NONE
        self.parse_buffer: str = ''
        self.skip: int = 0

        self.statement = None
        self.robot = robot
        self.selected_robot = None
        self.message_queue = message_queue

        self.character_count: int = 0

    def parse(self, output: Stream | str, stream_interpreting: bool=False) -> bool:
        for chunk in output:
            # Get the code from the chunk
            if isinstance(chunk, str):
                code = chunk
            else:
                code = chunk.choices[0].delta.content
            
            # Skip empty code
            if code == None or len(code) == 0:
                continue

            for c in code:
                self.character_count += 1
                match self.parse_state:
                    case ProgramParsingState.NONE:
                        self.statement = Statement({}, self.robot)
                        self.statement.parse('{')
                        self.statement.parse(c)
                        if self.message_queue:
                            self.message_queue.put(c + '\\\\')
                        self.parse_state = ProgramParsingState.PLAN

                    case ProgramParsingState.PLAN:
                        self.statement.parse(c, stream_interpreting)
                        if self.message_queue:
                            self.message_queue.put(c + '\\\\')
                        if self.character_count == len(output):
                            if self.statement.parse('}'):
                                # print(self.statement.to_string())
                                return True
                            else:
                                return False
                            
                            # # TODO: test executable
                            # if stream_interpreting and self.statement.executable and not self.statement.running:
                            #     # Send the statement to the execution queue
                            #     print(f'##### Statement executable: {self.statement.action}')
                            #     self.statement.running = True
                            #     Statement.execution_queue.put(self.statement)
        return False
    
    def _parse(self, json_output: Stream | str, stream_interpreting: bool=False) -> bool:
        for chunk in json_output:
            # Get the code from the chunk
            if isinstance(chunk, str):
                code = chunk
            else:
                code = chunk.choices[0].delta.content
            
            # Skip empty code
            if code == None or len(code) == 0:
                continue

            for c in code:
                if self.skip > 0:
                    self.skip -= 1
                    continue
                match self.parse_state:
                    case ProgramParsingState.NONE:
                        if c == '{':
                            self.parse_state = ProgramParsingState.JSON_BEGIN

                    case ProgramParsingState.JSON_BEGIN:
                        if c == '"':
                            self.parse_buffer = ""
                            self.parse_state = ProgramParsingState.PREFIX

                    # match for LLM_PLAN_START_PREFIX
                    case ProgramParsingState.PREFIX:
                        self.parse_buffer += c
                        if self.parse_buffer == "plan":
                            self.parse_buffer = ''
                            self.parse_state = ProgramParsingState.QUOTATION_START
                            self.skip = 1
                        elif not "plan".startswith(self.parse_buffer):
                            self.parse_state = ProgramParsingState.JSON_BEGIN

                    case ProgramParsingState.QUOTATION_START:
                        if c == ':':
                            self.parse_buffer = c
                        elif self.parse_buffer == ':' and c == '"':
                            self.parse_buffer = ''
                            self.parse_state = ProgramParsingState.PLAN
                            self.statement = Statement({}, self.robot)
                            self.statement.parse('{')
                        else:
                            continue

                    case ProgramParsingState.PLAN:
                        if c == '"':
                            self.parse_state = ProgramParsingState.QUOTATION_END
                            if self.statement.parse('}'):
                                # print(self.statement.to_string())
                                return True
                        else:
                            if self.message_queue:
                                self.message_queue.put(c + '\\\\')

                            self.statement.parse(c, stream_interpreting)
                            # # TODO: test executable
                            # if stream_interpreting and self.statement.executable and not self.statement.running:
                            #     # Send the statement to the execution queue
                            #     print(f'##### Statement executable: {self.statement.action}')
                            #     self.statement.running = True
                            #     Statement.execution_queue.put(self.statement)

                    case ProgramParsingState.QUOTATION_END:
                        if c == '}':
                            self.parse_state = ProgramParsingState.JSON_END
                    
                    case ProgramParsingState.JSON_END:
                        return False
        return False
    
    def eval(self) -> SKILL_RET_TYPE:
        return self.statement.eval()
    
    def stop(self):
        self.statement.stop()
    
class CodeAction(Enum):
    NONE = auto()
    ATOMIC = auto() # single statement
    SEQ = auto()    # sequence of statements
    IF = auto()     # if statement
    LOOP = auto()   # loop statement

class StatementParsingState(Enum):
    DEFAULT = auto()
    ARGUMENTS = auto()
    CONDITION = auto()
    LOOP_COUNT = auto()
    IF_SUB_STATEMENT = auto()
    ELSE_SUB_STATEMENT = auto()

class Statement:
    def __init__(self, env: dict, robot: RobotWrapper, log: bool=True):
        self.log = log

        self.parse_state = StatementParsingState.DEFAULT
        self.parse_buffer: str = ''
        self.parse_depth: int = 0

        self.action = CodeAction.NONE
        self.condition: list[str] = [] # `if`, `elif`, ...
        self.loop_count: int = 0
        self.current_statement = None

        # ATOMIC: single statement
        # SEQ: list of statements
        # IF: list of statements corresponding to conditions.
        #     The last statement is the `else` statement if len(sub_statements) > len(condition).
        # LOOP: single statement
        self.sub_statements: list[str | 'Statement'] = []
        
        self.allow_digit: bool = False
        self.quotation: bool = False

        self.executable: bool = False

        self.ret: bool = False
        self.env = env
        self.robot = robot

        # Add a pause event
        self.active_high_level_skill = None
        self.pause_event = Event()
        self.pause_event.set()  # Initially allow execution

        self.stop_event = Event()
        self.stop_event.clear()

    """
    Print the statement in a simple format for debugging.
    """
    def to_string_simple(self) -> str:
        s = ''
        if self.action == CodeAction.IF:
            s += f'IF {self.condition[0]} {{...}}'
        elif self.action == CodeAction.LOOP:
            s += f'[{self.loop_count}] {{...}}'
        elif self.action == CodeAction.SEQ:
            s += '{...}'
        elif self.action == CodeAction.ATOMIC:
            s += f'{self.sub_statements[0]}'
        else:
            raise Exception('Invalid action')
        
        return s
    
    def to_string_original(self) -> str:
        """
        Print the original statement.
        """
        s = ''
        if self.action == CodeAction.IF:
            s += f'? {self.condition[0]} {{{self.sub_statements[0].to_string_original()}}}'
        elif self.action == CodeAction.LOOP:
            s += f'[{self.loop_count}] {{{self.sub_statements[0].to_string_original()}}}'
        elif self.action == CodeAction.SEQ:
            for statement in self.sub_statements:
                if isinstance(statement, str):
                    s += f'{statement}; '
                else:
                    s += f'{statement.to_string_original()}; '
        elif self.action == CodeAction.ATOMIC:
            s += f'{self.sub_statements[0]}'
        else:
            raise Exception('Invalid action')
        
        return s

    """
    Print the full statement in a human-readable format for debugging.
    """
    def to_string(self, depth: int=0) -> str:
        indent = '****'
        prefix = indent * depth
        s = ''
        if self.action == CodeAction.IF:
            len1 = len(self.condition)
            len2 = len(self.sub_statements)
            
            for i in range(len1):
                if i > 0:
                    s += ' else '
                else:
                    s += prefix
                s += f'if {self.condition[i]}\n'
                s += f'{self.sub_statements[i].to_string(depth)}'

            if len2 > len1:
                s += ' else\n'
                s += self.sub_statements[-1].to_string(depth)

        elif self.action == CodeAction.LOOP:
            s += prefix + f'[{self.loop_count}]\n'
            s += self.sub_statements[0].to_string(depth)
        elif self.action == CodeAction.SEQ:
            s += prefix + '{\n'
            for statement in self.sub_statements:
                s += f'{statement.to_string(depth + 1)};\n'
            s += prefix + '}'
        elif self.action == CodeAction.ATOMIC:
            if not isinstance(self.sub_statements[0], str):
                raise Exception('Invalid action')
            s += prefix + self.sub_statements[0]
        else:
            raise Exception('Invalid action')
        
        return s
    
    def _get_env(self, var) -> SKILL_RET_TYPE:
        if var not in self.env:
            raise Exception(f'Variable {var} is not defined')
        return self.env[var]

    def parse(self, code: str, exec: bool = False) -> bool:
        for c in code:
            if c == ' ' and not self.quotation:
                continue
            # print('--' * self.depth + f'-> {c}, action: {self.action}, state: {self.parse_state}')

            if c == '\'' or c == '"':
                self.quotation = not self.quotation

            match self.action:
                case CodeAction.NONE:
                    if c == '{':
                        self.action = CodeAction.SEQ
                        self.current_statement = Statement(self.env, self.robot, self.log)
                        self.parse_depth += 1
                    elif c == '?':
                        self.action = CodeAction.IF
                        self.parse_buffer = ''
                        self.parse_state = StatementParsingState.CONDITION
                    elif c == ';' or c == '}':
                        return False
                    elif c.isalpha() or c == '_' or c == '-':
                        self.parse_buffer = c
                        self.action = CodeAction.ATOMIC
                        self.allow_digit = True
                    elif c.isdigit() and not self.allow_digit:
                        self.action = CodeAction.LOOP
                        self.parse_state = StatementParsingState.LOOP_COUNT
                        self.parse_buffer += c
                    else:
                        raise Exception(f'Invalid character: {c}')

                case CodeAction.ATOMIC:
                    if not self.quotation and (c == ';' or c == '}'):
                        self.sub_statements.append(self.parse_buffer)
                        self.executable = True
                        return True
                    else:
                        self.parse_buffer += c

                case CodeAction.SEQ:
                    done = self.current_statement.parse(c)

                    if self.current_statement.executable:
                        self.executable = True

                    if done:
                        self.sub_statements.append(self.current_statement)
                        self.current_statement = Statement(self.env, self.robot, self.log)
                        self.current_statement.parse(c)

                    if self.quotation:
                        continue

                    if c == '{':
                        self.parse_depth += 1
                    elif c == '}':
                        self.parse_depth -= 1
                        if self.parse_depth == 0:
                            return True

                case CodeAction.IF:
                    match self.parse_state:
                        case StatementParsingState.DEFAULT:
                            if c != ':':
                                return True
                            else:
                                self.parse_state = StatementParsingState.ELSE_SUB_STATEMENT
                        case StatementParsingState.CONDITION:
                            if c == '{' and not self.quotation:
                                self.condition.append(self.parse_buffer)
                                self.executable = True
                                self.parse_state = StatementParsingState.IF_SUB_STATEMENT
                                self.current_statement = Statement(self.env, self.robot, self.log)
                                self.current_statement.parse(c)
                                self.parse_depth += 1
                            else:
                                # read condition between '?' and '{'
                                self.parse_buffer += c
                        case StatementParsingState.IF_SUB_STATEMENT:
                            if self.current_statement.parse(c):
                                self.sub_statements.append(self.current_statement)
                                self.current_statement = Statement(self.env, self.robot, self.log)

                            if c == '{':
                                self.parse_depth += 1
                            elif c == '}':
                                self.parse_depth -= 1
                                if self.parse_depth == 0:
                                    self.parse_state = StatementParsingState.DEFAULT
                        case StatementParsingState.ELSE_SUB_STATEMENT:
                            if c == '?':
                                self.parse_buffer = ''
                                self.parse_state = StatementParsingState.CONDITION
                            elif c == '{':
                                self.current_statement = Statement(self.env, self.robot, self.log)
                                self.current_statement.parse(c)
                                self.parse_state = StatementParsingState.IF_SUB_STATEMENT
                                self.parse_depth += 1
                            else:
                                raise Exception(f'Invalid character: {c}')

                case CodeAction.LOOP:
                    match self.parse_state:
                        case StatementParsingState.LOOP_COUNT:
                            if c == '{':
                                self.loop_count = int(self.parse_buffer)
                                self.parse_buffer = ''
                                self.parse_state = StatementParsingState.DEFAULT
                                self.current_statement = Statement(self.env, self.robot, self.log)
                                self.current_statement.parse(c)
                            elif c.isdigit():
                                self.parse_buffer += c
                            else:
                                raise Exception(f'Invalid loop count: {self.parse_buffer}')
                        case StatementParsingState.DEFAULT:
                            done = self.current_statement.parse(c)
                            if self.current_statement.executable:
                                self.executable = True

                            if done:
                                self.sub_statements.append(self.current_statement)
                                return True
                            
        return False
    
    def pause(self):
        """Pause the execution of the statement."""
        print_t(f'Pause statement: {self.to_string_simple()}')
        self.pause_event.clear()

        if self.active_high_level_skill:
            self.active_high_level_skill.pause()

        for sub_statement in self.sub_statements:
            if isinstance(sub_statement, Statement):
                sub_statement.pause()

    def resume(self):
        """Resume the execution of the statement."""
        self.pause_event.set()

        if self.active_high_level_skill:
            self.active_high_level_skill.resume()

        for sub_statement in self.sub_statements:
            if isinstance(sub_statement, Statement):
                sub_statement.resume()

    def stop(self):
        """Stop the execution of the statement."""
        print_t(f'Stop statement: {self.to_string_simple()}')
        self.stop_event.set()

        if self.active_high_level_skill:
            self.active_high_level_skill.stop()

        for sub_statement in self.sub_statements:
            if isinstance(sub_statement, Statement):
                sub_statement.stop()

    def _check_pause_stop(self) -> bool:
        """Wait for pause to clear, return True if stop is set."""
        self.pause_event.wait()
        if self.stop_event.is_set():
            _print_debug(f'Stop statement: {self.to_string_simple()}')
            self.ret = True
            return True
        return False

    def eval(self) -> SKILL_RET_TYPE:
        _print_debug(f'Eval statement: {self.to_string()}')
        while not self.executable:
            time.sleep(0.1)
        default_ret_val: SKILL_RET_TYPE = None

        # Check if the statement is paused
        if self._check_pause_stop():
            return False

        match self.action:
            case CodeAction.ATOMIC:
                assert len(self.sub_statements) == 1 and isinstance(self.sub_statements[0], str)
                return self.eval_expr(self.sub_statements[0])

            case CodeAction.SEQ:
                for statement in self.sub_statements:
                    default_ret_val = statement.eval()
                    if statement.ret:
                        self.ret = True
                        return default_ret_val

            case CodeAction.IF:
                for i in range(len(self.condition)):
                    if self._check_pause_stop():
                        return False
                    if self.eval_condition(self.condition[i]) == True:
                        ret_val = self.sub_statements[i].eval()
                        if self.sub_statements[i].ret:
                            self.ret = True
                        return ret_val
                
                if len(self.sub_statements) > len(self.condition):
                    ret_val = self.sub_statements[-1].eval()
                    if self.sub_statements[-1].ret:
                        self.ret = True
                    return ret_val

            case CodeAction.LOOP:
                assert len(self.sub_statements) == 1
                for i in range(self.loop_count):
                    if self._check_pause_stop():
                        return False
                    ret_val = self.sub_statements[0].eval()
                    if self.sub_statements[0].ret:
                        self.ret = True
                        return ret_val

            case CodeAction.NONE:
                raise Exception('Invalid action')
        return default_ret_val

    def eval_function(self, func: str) -> SKILL_RET_TYPE:
        if self._check_pause_stop():
            return False
        _print_debug(f'Eval function: {func}')

        lp = func.find('(')
        rp = func.rfind(')')
        if lp == -1 or rp == -1 or lp > rp:
            raise Exception(f'Invalid function call: {func}')

        func_name = func[:lp].strip()
        args = func[lp + 1:rp].strip()

        if len(args) == 0:
            args = []
        else:
            args = re.split(r",\s*(?![^()]*\))", args)

        for i in range(len(args)):
            args[i] = self.eval_expr(args[i])
            
        # print(func_name, args)

        ll_skill = self.robot.ll_skillset.get_skill(func_name)
        if ll_skill:
            if self.log:
                self.robot.memory.get_subtask().append_action(func)
            rslt = ll_skill.execute(args)
            if self.log:
                self.robot.memory.get_subtask().finish_action(rslt != False)
            _print_debug(f'Executing low-level skill: {ll_skill.name} {args} {rslt}')
            return rslt

        hl_skill = self.robot.hl_skillset.get_skill(func_name)
        if hl_skill:
            _print_debug(f'Executing high-level skill: {hl_skill.name}', args, hl_skill.execute(args)[0])
            if self.log:
                self.robot.memory.get_subtask().append_action(func)
            self.active_high_level_skill = Statement(self.env, self.robot, False)
            self.active_high_level_skill.parse(hl_skill.execute(args))
            try:
                val = self.active_high_level_skill.eval()
            except Exception as e:
                print_t(f'Error executing high-level skill: {e}')
                val = False
            if self.log:
                self.robot.memory.get_subtask().finish_action(val != False)
            return val
        
        raise Exception(f'Skill {func_name} is not defined')

    def eval_expr(self, expr: str) -> SKILL_RET_TYPE:
        if self._check_pause_stop():
            return False
        _print_debug(f'Eval expr: {expr}')
        expr = expr.strip()

        # stripe ()
        if expr.startswith('(') and expr.endswith(')'):
            expr = expr[1:-1].strip()

        if len(expr) == 0:
            raise Exception('Empty expression')
        
        # Handle return value (->)
        if expr.startswith('->'):
            self.ret = True
            return self.eval_expr(expr.lstrip('->'))
        
        # Handle variable assignment (_var = ...)
        if expr.startswith('_') and '=' in expr:
            var, expr = expr.split('=', 1)
            var = var.strip()
            _print_debug(f'Eval expr var assign: {var}={expr}')
            ret_val = self.eval_expr(expr)
            _print_debug(f'==> var assign: {var}={ret_val}')
            self.env[var] = ret_val
            return ret_val
        
        # Handle arithmetic operations
        operators = {
            '+': lambda a, b: a + b,
            '-': lambda a, b: a - b,
            '*': lambda a, b: a * b,
            '/': lambda a, b: a / b,
        }

        # Split the expression while respecting parentheses and negative numbers
        def split_expression(expr, operator):
            depth = 0
            parts = []
            current = []
            for i, char in enumerate(expr):
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
                elif char == operator and depth == 0:
                    # Ensure we don't split on a negative sign (e.g., "-10")
                    if operator == '-' and (i == 0 or expr[i - 1] in ' ('):
                        current.append(char)
                        continue
                    parts.append(''.join(current).strip())
                    current = []
                    continue
                current.append(char)
            parts.append(''.join(current).strip())
            return parts

        for op, func in operators.items():
            if op in expr:
                operands = split_expression(expr, op)
                print(f'Operands: {operands}')
                if len(operands) < 2:
                    continue
                # Evaluate the first operand
                result = self.eval_expr(operands[0])
                # Apply the operator to the remaining operands
                for operand in operands[1:]:
                    result = func(result, self.eval_expr(operand.strip()))
                return result

        # Handle variables, constants, and function calls
        if expr.startswith('_'):
            return self._get_env(expr)
        elif expr == 'True' or expr == 'False':
            return evaluate_value(expr)
        elif expr[0].isalpha():
            return self.eval_function(expr)
        else:
            return evaluate_value(expr)

    def eval_condition(self, condition: str) -> SKILL_RET_TYPE:
        if self._check_pause_stop():
            return False
        ### TODO: add support for nested conditions

        # Multiple conditions
        if '&&' in condition:
            conditions = condition.split('&&')
            for c in conditions:
                if self.eval_condition(c) == False:
                    return False
            return True
        if '||' in condition:
            conditions = condition.split('||')
            for c in conditions:
                if self.eval_condition(c) == True:
                    return True
            return False
        
        # Single condition
        parts = re.split(r'(>|<|==|!=)', condition)
        if len(parts) != 1 and len(parts) != 3:
            raise Exception(f'Invalid condition format: {condition}')

        operand_1 = parts[0]
        operand_1 = self.eval_expr(operand_1)

        if len(parts) == 3:
            comparator, operand_2 = parts[1], parts[2]
            operand_2 = self.eval_expr(operand_2)
            _print_debug(f'Condition ops: {operand_1} {comparator} {operand_2}')
        else:
            _print_debug(f'Condition ops: {operand_1}')
            return operand_1 != False

        if isinstance(operand_1, (int, float)) and isinstance(operand_2, (int, float)):
            operand_1 = float(operand_1)
            operand_2 = float(operand_2)

        if type(operand_1) != type(operand_2):
            if comparator == '!=':
                return True
            elif comparator == '==':
                return False
            else:
                raise Exception(f'Invalid comparator: {operand_1}:{type(operand_1)} {operand_2}:{type(operand_2)}')

        cmp = None
        if comparator == '>':
            cmp = operand_1 > operand_2
        elif comparator == '<':
            cmp = operand_1 < operand_2
        elif comparator == '==':
            cmp = operand_1 == operand_2
        elif comparator == '!=':
            cmp = operand_1 != operand_2

        if cmp is None:
            raise Exception(f'Invalid comparator: {comparator}')
        return cmp

class MiniSpecInterpreter:
    def __init__(self, message_queue: queue.Queue, robot: RobotWrapper):
        self.robot = robot
        self.message_queue = message_queue

        self.execution_history = []

        self.program = None
        self.exit = False

        self.execution_thread = Thread(target=self.executor)
        self.execution_thread.start()

    def execute(self, json_output: Stream | str) -> SKILL_RET_TYPE:
        self.execution_history = []
        self.timestamp_get_plan = time.time()

        stream_interpreting = False if isinstance(json_output, str) else True

        if stream_interpreting:
            self.execution_thread.start()

        self.program = MiniSpecProgram(self.robot, self.message_queue)
        self.program.parse(json_output, stream_interpreting)

        if stream_interpreting:
            print_t(f"[M] Program received in {time.time() - self.timestamp_get_plan}s")
        else:
            print_t("[M] Start normal execution")

    def executor(self):
        while True:
            if self.program and self.program.statement and self.program.statement.executable:
                print_t("[M] Start execution")
                self.program.eval()
                self.exit = True
                return
            else:
                time.sleep(0.005)