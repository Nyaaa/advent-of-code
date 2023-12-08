import re
from itertools import cycle

from tools import loader, parsers


def desert(data: list[list[str]]) -> int:
    """
    >>> print(desert(parsers.blocks('test.txt')))
    6"""
    instructions = cycle(data[0][0])
    mapping = {k: re.findall(r'\w+', v) for k, v in (i.split(' = ') for i in data[1])}
    location = 'AAA'
    steps = 0
    while location != 'ZZZ':
        turn = next(instructions)
        location = mapping[location][0 if turn == 'L' else 1]
        steps += 1
    return steps


print(desert(parsers.blocks(loader.get())))  # 19667
