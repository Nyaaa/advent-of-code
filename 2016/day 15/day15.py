from itertools import count
from tools import parsers, loader
import re

TEST = """Disc #1 has 5 positions; at time=0, it is at position 4.
Disc #2 has 2 positions; at time=0, it is at position 1."""


def disks(data: list[str], part2: bool) -> int:
    """
    >>> print(disks(parsers.inline_test(TEST), False))
    5"""
    col = [tuple(map(int, re.findall(r'\d+', line))) for line in data]
    if part2:
        col.append((len(col) + 1, 11, 0, 0))
    for time in count():
        for index, positions, _, current in col:
            pos = (current + time + index) % positions
            if pos != 0:
                break
        else:
            return time
    raise ValueError('No solution found.')


print(disks(parsers.lines(loader.get()), False))  # 16824
print(disks(parsers.lines(loader.get()), True))  # 3543984
