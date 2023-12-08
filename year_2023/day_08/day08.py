import re
from itertools import cycle
from math import lcm

from tools import loader, parsers


def desert(data: list[list[str]], part2: bool) -> int:
    """
    >>> print(desert(parsers.blocks('test.txt'), part2=False))
    6
    >>> print(desert(parsers.blocks('test2.txt'), part2=True))
    6"""
    def move(location: str) -> int:
        instructions = cycle(data[0][0])
        steps = 0
        while not location.endswith('Z'):
            location = mapping[location][0 if next(instructions) == 'L' else 1]
            steps += 1
        return steps

    mapping = {k: re.findall(r'\w+', v) for k, v in (i.split(' = ') for i in data[1])}
    starts = ['AAA'] if not part2 else re.findall(r'(..A)\s', ' '.join(data[1]))
    return lcm(*(move(start) for start in starts))


print(desert(parsers.blocks(loader.get()), part2=False))  # 19667
print(desert(parsers.blocks(loader.get()), part2=True))  # 19185263738117
