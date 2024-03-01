from ast import literal_eval
from functools import cmp_to_key
from itertools import zip_longest

from tools import loader, parsers


def compare(left: list, right: list) -> int:
    if not isinstance(left, list): left = [left]
    if not isinstance(right, list): right = [right]

    for i, j in zip_longest(left, right):
        match i, j:
            case _, None:
                return 1
            case None, _:
                return -1
            case int(), int():
                if i != j:
                    return i - j
            case _:
                out = compare(i, j)
                if out != 0:
                    return out
    return 0


def part_1(data: list[list[str]]) -> int:
    """test part 1:
    >>> part_1(parsers.blocks('test13.txt'))
    13"""
    result = 0
    for index, pair in enumerate(data, start=1):
        if compare(*map(literal_eval, pair)) < 0:
            result += index
    return result


def part_2(data: list[str]) -> int:
    """test part 2:
    >>> part_2(parsers.lines('test13.txt'))
    140"""
    data.extend(('[[2]]', '[[6]]'))
    data = [literal_eval(line) for line in data if line]
    data.sort(key=cmp_to_key(compare))
    return (data.index([[2]]) + 1) * (data.index([[6]]) + 1)


print(part_1(parsers.blocks(loader.get())))  # 5252
print(part_2(parsers.lines(loader.get())))  # 20592
