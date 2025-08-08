# tiny timing helper to measure a code block with a context manager
import time
from contextlib import contextmanager

@contextmanager
def timer(name: str, timings: dict | None = None):
    t0 = time.time()
    try:
        yield
    finally:
        dt = time.time() - t0
        if isinstance(timings, dict):
            timings[name] = dt  # shove it in the dict so caller can read later
