import logging
import sys
import time
from functools import wraps

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)


class context:
    """A timer context manager.

    Usage:

    with timer.context():
        some_code
    """
    def __enter__(self):
        self.start = time.perf_counter()

    def __exit__(self, *args):
        end = time.perf_counter()
        logging.debug(f'Code executed in {end - self.start:.8f} seconds.')


def wrapper(func):
    """A timer wrapper.

    Usage:

    @timer.wrapper
    def some_func():
    """
    @wraps(func)
    def wrapped(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        func_name = func.__name__
        logging.debug(f'{func_name} executed in {end - start:.8f} seconds.')
        return result
    return wrapped
