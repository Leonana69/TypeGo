import threading
import functools
from typing import Callable, Iterable, Optional, Any, Dict

def auto_locked_properties(
    *,
    fields: Optional[Iterable[str]] = None,               # explicitly list private fields (optional)
    copy_on_get: Iterable[str] = (),                      # property names to .copy() on get (e.g. numpy arrays, PIL images)
    set_cast: Dict[str, Callable[[Any], Any]] = None,     # property -> cast(fn) applied on set
):
    """
    Creates, for each private field '_foo':
      - a lock named '_foo_lock' (per-instance)
      - a property 'foo' whose getter/setter are guarded by that lock
    By default it discovers fields from class-level __annotations__ that startwith '_'.
    """
    copy_on_get = set(copy_on_get)
    set_cast = set_cast or {}

    def decorator(cls):
        # Discover private fields
        privs = []
        if fields is not None:
            privs = [n for n in fields if n.startswith('_')]
        else:
            anns = getattr(cls, '__annotations__', {})
            privs = [n for n in anns.keys() if isinstance(n, str) and n.startswith('_')]

        # Add properties for each private field
        for priv in privs:
            public = priv[1:]                    # '_image' -> 'image'
            lock_name = f"{priv}_lock"           # '_image_lock'

            # create getter/setter with default-arg binding to avoid late-binding bug
            def getter(self, _priv=priv, _ln=lock_name, _pub=public):
                lock = getattr(self, _ln)
                with lock:
                    val = getattr(self, _priv)
                    if _pub in copy_on_get and hasattr(val, "copy"):
                        try:
                            return val.copy()
                        except Exception:
                            pass
                    return val

            def setter(self, value, _priv=priv, _ln=lock_name, _pub=public):
                if _pub in set_cast:
                    value = set_cast[_pub](value)
                lock = getattr(self, _ln)
                with lock:
                    setattr(self, _priv, value)

            setattr(cls, public, property(getter, setter))

            # Propagate type hint to the public property (helps editors/linters)
            anns = getattr(cls, '__annotations__', None)
            if isinstance(anns, dict) and priv in anns and public not in anns:
                anns[public] = anns[priv]

        # Wrap __init__ to create locks (after your original __init__)
        orig_init = cls.__init__

        @functools.wraps(orig_init)
        def __init__(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            for priv in privs:
                ln = f"{priv}_lock"
                if not hasattr(self, ln):
                    setattr(self, ln, threading.Lock())

        cls.__init__ = __init__
        return cls
    return decorator
