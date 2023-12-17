from collections import Counter
from functools import cache
from itertools import pairwise

from tools import loader, parsers


def prep(data: list) -> tuple[int, ...]:
    data = sorted(int(i) for i in data)
    joltages = [0, *data, data[-1] + 3]
    return tuple(joltages)


def part_1(joltages: tuple[int, ...]) -> int:
    """ test part 1:
    >>> print(part_1(TEST))
    220"""
    pairs = [j - i for i, j in pairwise(joltages)]
    counter = Counter(pairs)
    return counter[1] * counter[3]


@cache
def part_2(joltages: tuple[int, ...], item: int | None = None) -> int:
    """ test part 2:
    >>> print(part_2(TEST))
    19208"""
    if item is None:
        item = joltages[-1]
    elif item == 0:
        return 1
    candidates = [i for i in range(item - 3, item) if i in joltages]
    return sum(part_2(joltages, i) for i in candidates)


TEST = prep(parsers.lines('test.txt'))
DATA = prep(parsers.lines(loader.get()))
print(part_1(DATA))  # 2760
print(part_2(DATA))  # 13816758796288
