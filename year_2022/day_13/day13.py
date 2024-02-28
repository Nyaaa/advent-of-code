from ast import literal_eval
from itertools import zip_longest

from tools import loader, parsers


def compare(left: list, right: list) -> bool:
    if not isinstance(left, list): left = [left]
    if not isinstance(right, list): right = [right]

    zipped = list(zip_longest(left, right))
    for i, j in zipped:
        if j is None:
            return False
        if i is None:
            return True

        if isinstance(i, int) and isinstance(j, int):
            if i != j:
                return i < j
        else:
            out = compare(i, j)
            if out:
                return out
    return False


def flatten(list_of_lists: list) -> list:
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        if len(list_of_lists[0]) == 0:
            list_of_lists[0] = [0]
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])


def part_1(data: list[list[str]]) -> int:
    """test part 1:
    >>> part_1(parsers.blocks('test13.txt'))
    13"""
    result = 0
    for index, pair in enumerate(data):
        c_left, c_right = map(literal_eval, pair)
        if compare(c_left, c_right):
            result += index + 1
    return result


def part_2(data: list[str]) -> int:
    """test part 2:
    >>> part_2(parsers.lines('test13.txt'))
    140"""
    data.extend(('[[2]]', '[[6]]'))
    sort = [flatten(literal_eval(line)) for line in data if line]
    sort.sort()
    return (sort.index([2]) + 1) * (sort.index([6]) + 1)


print(part_1(parsers.blocks(loader.get())))  # 5252
print(part_2(parsers.lines(loader.get())))  # 20592
