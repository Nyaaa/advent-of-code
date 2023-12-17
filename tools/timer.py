import logging
import sys
import time
from collections.abc import Callable
from functools import wraps
from types import TracebackType
from typing import ParamSpec, TypeVar

logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
T = TypeVar('T')
P = ParamSpec('P')
C = TypeVar('C', bound=Callable)


class Context:
    """A timer context manager.

    Usage:

    with timer.Context():
        some_code
    """

    def __enter__(self) -> None:
        self.start = time.perf_counter()

    def __exit__(self,
                 exc_type: type[BaseException] | None,
                 exc_val: BaseException | None,
                 exc_tb: TracebackType | None) -> None:
        end = time.perf_counter()
        logging.debug(f'Code executed in {end - self.start:.8f} seconds.')


def wrapper(func: C) -> C:
    """A timer decorator.

    Usage:

    @timer.wrapper
    def some_func():
    """

    @wraps(func)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        func_name = func.__name__
        logging.debug(f'{func_name} executed in {end - start:.8f} seconds.')
        return result

    return wrapped
