from collections import Counter
from itertools import pairwise
from tools import parsers, loader


def part_1(data: list):
    """ test part 1:
    >>> print(part_1(parsers.lines('test.txt')))
    220"""
    data = sorted(int(i) for i in data)
    joltages = [0] + data + [data[-1] + 3]
    pairs = [j - i for i, j in pairwise(joltages)]
    counter = Counter(pairs)
    return counter[1] * counter[3]


print(part_1(parsers.lines(loader.get())))  # 2760
