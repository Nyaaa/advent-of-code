from collections.abc import Generator, Callable
from tools import parsers, loader
from numba import njit


TEST = """Generator A starts with 65
Generator B starts with 8921"""


@njit
def gen_part_1(previous: int, factor: int) -> Generator[int]:
    while True:
        previous = previous * factor % 2147483647
        yield previous


@njit
def gen_part_2(previous: int, factor: int) -> Generator[int]:
    mul = 4 if factor == 16807 else 8
    while True:
        previous = previous * factor % 2147483647
        if previous % mul == 0:
            yield previous


@njit
def duel(a: int, b: int, gen_func: Callable[Generator], limit: int) -> int:
    gen_a = gen_func(a, 16807)
    gen_b = gen_func(b, 48271)
    counter = 0
    step = 0
    while step < limit:
        step += 1
        a = next(gen_a)
        b = next(gen_b)
        if a & 0xFFFF == b & 0xFFFF:
            counter += 1
    return counter


def start(data: list[str], limit: int, part2: bool) -> int:
    """Numba can't convert strings to integers.

    >>> print(start(parsers.inline_test(TEST), 5, False))
    1

    >>> print(start(parsers.inline_test(TEST), 5_000_000, True))
    309"""
    a = int(data[0].split()[-1])
    b = int(data[1].split()[-1])
    gen_func = gen_part_2 if part2 else gen_part_1
    return duel(a, b, gen_func, limit)


print(start(parsers.lines(loader.get()), 40_000_000, False))  # 626
print(start(parsers.lines(loader.get()), 5_000_000, True))  # 306
