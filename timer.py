# v9 utils/timer.py — super tiny timing context manager
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
            timings[name] = dt
