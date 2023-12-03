import re
from collections import defaultdict
from itertools import accumulate
from math import prod

import numpy as np

from tools import common, loader, parsers


def parts(data: list[str]) -> tuple[int, int]:
    """
    >>> print(parts(parsers.lines('test.txt')))
    (4361, 467835)"""
    arr = np.asarray([list(line) for line in data], dtype=str)
    nums = [list(re.finditer(r'\d+', line)) for line in data]
    part1 = 0
    gears = defaultdict(set)
    for i, matches in enumerate(nums):
        for num in matches:
            is_valid = False
            for j in range(num.start(), num.end()):
                for loc, char in common.get_adjacent(arr, (i, j), with_corners=True):
                    if not is_valid and not char.isalnum() and char != '.':
                        is_valid = True
                    if char == '*':
                        gears[loc].add(int(num.group()))
            if is_valid:
                part1 += int(num.group())
    *_, part2 = accumulate(prod(p) for p in gears.values() if len(p) > 1)
    return part1, part2


print(parts(parsers.lines(loader.get())))  # 560670, 91622824
