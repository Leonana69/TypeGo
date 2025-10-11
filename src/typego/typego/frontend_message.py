import queue, io, base64
from typing import Any, Optional
from PIL import Image
from typego.utils import print_t

# Dictionary to store queues per task_id
_TASK_QUEUES: dict[int, "queue.Queue[Any]"] = {}


def get_queue(task_id: int, create_if_missing: bool = True) -> Optional["queue.Queue[Any]"]:
    """Return (and create if needed) the message queue for a specific task_id."""
    if task_id not in _TASK_QUEUES:
        if create_if_missing:
            _TASK_QUEUES[task_id] = queue.Queue()
            print_t(f"[QUEUE] Created queue for task {task_id}")
        else:
            return None
    return _TASK_QUEUES[task_id]


def publish(msg: Any, task_id: int) -> bool:
    """Put a message on the queue associated with the given task_id."""
    q = get_queue(task_id)

    if isinstance(msg, Image.Image):
        buffer = io.BytesIO()
        msg.save(buffer, format="JPEG")
        encoded_img = base64.b64encode(buffer.getvalue()).decode("utf-8")
        q.put(f'<img src="data:image/jpeg;base64,{encoded_img}" />')
    else:
        text = msg.strip("'")
        formatted_msg = f"[TASK_ID] {task_id}: {text} "
        q.put(formatted_msg)
        print_t(formatted_msg)
    return True


def try_get(timeout: float, task_id: int) -> Optional[Any]:
    """Attempt to get a message from the queue for the given task_id."""
    q = get_queue(task_id, create_if_missing=False)
    if not q:
        return None
    try:
        return q.get(timeout=timeout)
    except queue.Empty:
        return ""


def end_queue(task_id: int) -> None:
    """Clean up and remove the queue for a finished task."""
    if task_id in _TASK_QUEUES:
        q = _TASK_QUEUES.pop(task_id)
        # Drain remaining messages (optional)
        while not q.empty():
            q.get_nowait()
        print_t(f"[QUEUE] Queue for task {task_id} removed.")
    else:
        print_t(f"[QUEUE] No queue found for task {task_id}.")
