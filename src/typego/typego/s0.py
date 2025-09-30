from random import random
import time
import threading
from typing import Optional
import queue

from typego.robot_wrapper import RobotWrapper
from typego.utils import print_t

class S0Event:
    def __init__(self, description: str, condition: callable,
                 actions: list[callable], priority: int = 0, min_interval: float = 3.0):
        self.description = description
        self.condition = condition
        self.actions = actions
        self.min_interval = min_interval
        self.priority = priority
        self.timer = time.time()

    def check(self, current_priority: int = 99) -> bool:
        if self.priority < current_priority and self.condition() and (time.time() - self.timer) > 0.0:
            self.timer = time.time() + self.min_interval
            return True
        return False

    def execute(self, registry, on_finished: Optional[callable] = None):
        """Non-blocking execution: starts actions in a background thread.
        Calls on_finished() when all actions complete."""
        print_t(f"[S0Event] Executing {self.description}")

        def run_actions():
            exec_ids = []
            for action in self.actions:
                result = action()
                # If registry skill, capture exec_id
                print_t(f"[S0Event] Action result: {result}")

                if isinstance(result, dict) and result.get("ok") and "id" in result:
                    exec_ids.append(result["id"])
            # Wait for all registry executions to finish
            for eid in exec_ids:
                while True:
                    status = registry.get_status(eid)
                    if not status["ok"] or not status["alive"]:
                        break
                    time.sleep(0.05)
            print_t(f"[S0Event] Finished {self.description}")
            if on_finished:
                on_finished()

        threading.Thread(target=run_actions, daemon=True).start()


class S0:
    def __init__(self, robot: RobotWrapper):
        self.robot = robot
        self.in_progress_event: Optional[S0Event] = None
        self.running = True

    def loop(self, rate: float = 100.0):
        delay = 1 / rate
        print_t(f"[C] Starting S0...")

        s0events = [
            S0Event("Step back",
                    lambda: self.robot.observation.blocked(),
                    [lambda: self.robot.registry.execute("move_back(0.3)")],
                    0, min_interval=1.0),
            # S0Event("Avoid person",
            #         lambda: self.robot.get_obj_info("person") is not None,
            #         [lambda: self.robot.registry.execute("turn_right(90)") if random() < 0.5 else self.robot.registry.execute("turn_left(90)")],
            #         5, min_interval=4.0),
        ]

        self.in_progress_event: Optional[S0Event] = None
        event_queue = queue.Queue()
        event_executor_thread = threading.Thread(target=self.event_executor,
                                                    args=(event_queue,))
        event_executor_thread.start()

        while self.running:
            for event in s0events:
                if event.check(self.in_progress_event.priority if self.in_progress_event else 99):
                    if self.in_progress_event is None:
                        print_t(f"[C] New event triggered: {event.description}")
                        event_queue.put(("normal", event))

                    elif event.priority < self.in_progress_event.priority:
                        print_t(f"[C] Preempting {self.in_progress_event.description} with {event.description}")
                        event_queue.put(("preempt", event))
            time.sleep(delay)

    def event_executor(self, event_queue: queue.Queue[tuple[str, "S0Event"]]):
        while self.running:
            try:
                mode, event = event_queue.get(timeout=1)

                def on_finished():
                    if mode == "normal":
                        self.robot.resume_action()
                    self.in_progress_event = None

                if mode == "preempt":
                    if self.in_progress_event:
                        self.robot.stop_action()
                    self.in_progress_event = event
                    event.execute(self.robot.registry, on_finished)

                elif mode == "normal":
                    self.robot.pause_action()
                    self.in_progress_event = event
                    event.execute(self.robot.registry, on_finished)

            except queue.Empty:
                continue