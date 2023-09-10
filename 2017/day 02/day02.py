from tools import parsers, loader
from itertools import permutations

TEST = """5 1 9 5
7 5 3
2 4 6 8"""
TEST2 = """5 9 2 8
9 4 7 3
3 8 6 5"""


def part_1(data: list[str]) -> int:
    """
    >>> print(part_1(parsers.inline_test(TEST)))
    18"""
    return sum(max(i) - min(i) for i in [[int(i) for i in row.split()] for row in data])


def part_2(data: list[str]) -> int:
    """
    >>> print(part_2(parsers.inline_test(TEST2)))
    9"""
    nums = [[int(i) for i in row.split()] for row in data]
    return sum(a // b if a % b == 0 else 0 for i in nums for a, b in permutations(i, 2))


print(part_1(parsers.lines(loader.get())))  # 36766
print(part_2(parsers.lines(loader.get())))  # 261
