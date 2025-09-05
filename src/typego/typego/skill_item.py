from abc import ABC
import re
from typing import Any, Optional, TypeAlias
from typego.utils import log_error

SKILL_ARG_TYPE: TypeAlias = int | float | str
SKILL_RET_TYPE: TypeAlias = Optional[int | float | bool | str]

class SkillRegistry:
    def __init__(self):
        self._items: dict[str, "SkillItem"] = {}
        self._funcs: dict[str, callable] = {}

    def register(self, name: str, description: str = "", params: dict = None):
        def deco(fn):
            item = SkillItem(name=name, description=description)
            item.register_args(params)
            self._items[name] = item
            self._funcs[name] = fn
            return fn
        return deco

    def names(self) -> list[str]:
        return list(self._items.keys())

    def get_skill_list(self, keys: list[str]=[]) -> list[str]:
        """Returns a list of target registered skills in string format."""
        if keys:
            l = []
            for key in keys:
                if key in self._items:
                    l.append(str(self._items[key]))
                else:
                    log_error(f"Skill '{key}' not found in registry.")
            return l
        return [str(item) for item in self._items.values()]

    def execute(
        self,
        func_call: str,
        args: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Execute a registered skill by name with typed args."""
        if not args:
            # Parse the function call
            match = re.match(r"(\w+)\((.*)\)", func_call)
            if not match:
                return {"ok": False, "error": f"invalid function call '{func_call}'"}

            name = match.group(1)
            arg_list = [arg.strip() for arg in match.group(2).split(",") if arg.strip()]
        else:
            name = func_call
            arg_list = list(args.values())

        print(f"[SkillRegistry] Executing skill '{name}' with args {arg_list}")

        item = self._items.get(name)
        if not item:
            return {"ok": False, "error": f"unknown skill '{name}'"}
        try:
            parsed = item.parse_args(arg_list)
            ret = self._funcs[name](*parsed)
            return {"ok": True, "data": ret}
        except Exception as e:
            return {"ok": False, "error": str(e)}

class SkillArg:
    def __init__(self, arg_name: str, arg_type: type):
        self.arg_name = arg_name
        self.arg_type = arg_type
    
    def __repr__(self) -> str:
        return f"{self.arg_name}: {self.arg_type.__name__}"

class SkillItem(ABC):
    def __init__(self, name: str, description: str):
        self._name: str = name
        self._description: str = description
        self._args: list[SkillArg] = []

    @property
    def name(self): return self._name
    @property
    def description(self): return self._description
    @property
    def args(self): return tuple(self._args)
    
    def __repr__(self) -> str:
        return (f"{self._name}: "
                f"args: {[arg for arg in self._args]}, "
                f"desc: {self._description}")

    def register_args(self, params: dict):
        for k, v in params.items():
            self._args.append(SkillArg(k, v))

    def parse_args(self, args_str_list: list[SKILL_ARG_TYPE], allow_positional_args: bool = False) -> list[SKILL_ARG_TYPE]:
        """Parses the string of arguments and converts them to the expected types."""
        # Check the number of arguments
        if len(args_str_list) != len(self.args):
            log_error(f"Func {self.name} expected {len(self.args)} arguments, but got {len(args_str_list)}.", raise_error=True)

        parsed_args = []
        for i, arg in enumerate(args_str_list):
            # if arg is not a string, skip parsing
            if not isinstance(arg, str):
                parsed_args.append(arg)
                continue
            # Allow positional arguments
            if arg.startswith('$') and allow_positional_args:
                parsed_args.append(arg)
                continue
            try:
                if self.args[i].arg_type == bool:
                    parsed_args.append(arg.strip().lower() == 'true')
                else:
                    parsed_args.append(self.args[i].arg_type(arg.strip()))
            except ValueError as e:
                log_error(f"Error parsing argument {i + 1}. Expected type {self._args[i].arg_type.__name__}, but got value '{arg.strip()}'. Original error: {e}", raise_error=True)
        return parsed_args
    

def evaluate_value(s: str) -> SKILL_RET_TYPE:
    if not s:  # Empty string or None
        return None
    
    # Strip whitespace once at the beginning
    s_clean = s.strip()
    
    # Check for None
    if s_clean == 'None':
        return None
    
    # Check for boolean values
    if s_clean == 'True':
        return True
    if s_clean == 'False':
        return False
    
    # Check for numeric values
    if s_clean.startswith(('-', '+')):
        num_str = s_clean[1:]
        sign = -1 if s_clean[0] == '-' else 1
    else:
        num_str = s_clean
        sign = 1
    
    # Check if it's a valid number
    if num_str.replace('.', '', 1).isdigit():
        if '.' in num_str:
            return sign * float(num_str)
        else:
            return sign * int(num_str)
    
    # Return original string if no conversion applies
    return s