from itertools import zip_longest
from tools import parsers, loader


def compare(left, right) -> bool:
    if not isinstance(left, list): left = [left]
    if not isinstance(right, list): right = [right]

    zipped = list(zip_longest(left, right))
    for i, j in zipped:
        if j is None:
            return False
        elif i is None:
            return True

        if isinstance(i, int) and isinstance(j, int):
            if i != j:
                return i < j
        else:
            out = compare(i, j)
            if out is not None:
                return out


def flatten(list_of_lists: list) -> list:
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        if len(list_of_lists[0]) == 0:
            list_of_lists[0] = [0]
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])


def part_1(data):
    """test part 1:
    >>> part_1(parsers.blocks('test13.txt'))
    13"""
    result = {True: [], False: []}
    for index in range(1, len(data) + 1):
        c_left, c_right = map(eval, data[index - 1])
        res = compare(c_left, c_right)
        result[res].append(index)
    return sum(result[True])


def part_2(data):
    """test part 2:
    >>> part_2(parsers.lines('test13.txt'))
    140"""
    data.append('[[2]]')
    data.append('[[6]]')
    sort = [flatten(eval(line)) for line in data if line != '']
    sort.sort()
    return (sort.index([2]) + 1) * (sort.index([6]) + 1)


print(part_1(parsers.blocks(loader.get())))  # 5252
print(part_2(parsers.lines(loader.get())))  # 20592
