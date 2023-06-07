from itertools import combinations
from tools import parsers, loader


def part_1(data: list, chunk: int = 25) -> int:
    """ test part 1:
    >>> print(part_1(parsers.lines('test.txt'), chunk=5))
    127"""
    data = [int(i) for i in data]
    for i, val in enumerate(data):
        if i < chunk:
            continue
        preamble = data[i-chunk:i]
        if val not in (sum(i) for i in combinations(preamble, 2)):
            return val
    raise ValueError('Number not found')


print(part_1(parsers.lines(loader.get())))  # 177777905
