# message_bus.py
import queue, io, base64
from typing import Any, Optional
from PIL import Image
from typego.utils import print_t

# Single, shared queue for the whole app
_MESSAGE_QUEUE: "queue.Queue[Any]" = queue.Queue()

def get_queue() -> "queue.Queue[Any]":
    """Return the global message queue."""
    return _MESSAGE_QUEUE

def publish(msg: Any) -> None:
    """Convenience: put a message on the global queue."""
    if isinstance(msg, Image.Image):
        buffer = io.BytesIO()
        msg.save(buffer, format="JPEG")
        _MESSAGE_QUEUE.put(f'<img src="data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode("utf-8")}" />')
    else:
        text = msg.strip('\'')
        _MESSAGE_QUEUE.put(f'[LOG] {text}')
        print_t(f'[LOG] {text}')
    return True

def try_get(timeout: float = 0.0) -> Optional[Any]:
    """Non-blocking/timeout read from the global queue; returns None on empty."""
    try:
        return _MESSAGE_QUEUE.get(timeout=timeout)
    except queue.Empty:
        return None