from collections import Counter

from tools import loader, parsers


def prep(data: list[str]) -> tuple[list[int], list[int]]:
    left = []
    right = []
    for i in data:
        lft, rgt = i.split()
        left.append(int(lft))
        right.append(int(rgt))

    return left, right


def part1(data: list[str]) -> int:
    """
    >>> print(part1(parsers.lines('test.txt')))
    11"""
    left, right = prep(data)
    left.sort()
    right.sort()
    return sum(abs(lft - rgt) for lft, rgt in zip(left, right, strict=False))


def part2(data: list[str]) -> int:
    """
    >>> print(part2(parsers.lines('test.txt')))
    31"""
    left, right = prep(data)
    right = Counter(right)
    return sum(num * right.get(num, 0) for num in left)


print(part1(parsers.lines(loader.get())))  # 2742123
print(part2(parsers.lines(loader.get())))  # 21328497
