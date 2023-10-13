from collections import Counter

from tools import loader, parsers


def count(data: list[list[str]]) -> tuple[int, int]:
    """
    >>> print(count(parsers.blocks('test.txt')))
    (11, 6)"""
    part_1 = 0
    part_2 = 0
    for block in data:
        letters = [i for line in block for i in list(line)]
        part_1 += len(set(letters))
        people = len(block)
        for i in Counter(letters).values():
            part_2 += 1 if i == people else 0
    return part_1, part_2


print(count(parsers.blocks(loader.get())))  # 6662, 3382
