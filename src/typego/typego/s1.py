from typego.plan_generator import PlanGenerator

class S1:
    def __init__(self, planner: PlanGenerator):
        self.planner = planner

    def plan(self, inst: str) -> str:
        return self.planner.s1_plan(inst)