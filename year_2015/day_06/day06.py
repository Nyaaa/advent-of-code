import re

import numpy as np

from tools import loader, parsers


def part_1(data: list[str]) -> int:
    grid = np.zeros((1000, 1000), dtype=bool)
    for line in data:
        nums = tuple(map(int, re.findall(r'\d+', line)))
        area = np.s_[nums[0]:nums[2] + 1, nums[1]:nums[3] + 1]
        if line.startswith('turn on'):
            grid[area] = True
        elif line.startswith('turn off'):
            grid[area] = False
        elif line.startswith('toggle'):
            grid[area] = np.invert(grid[area])
    return np.count_nonzero(grid)


def part_2(data: list[str]) -> int:
    grid = np.zeros((1000, 1000), dtype=int)
    for line in data:
        nums = tuple(map(int, re.findall(r'\d+', line)))
        area = np.s_[nums[0]:nums[2] + 1, nums[1]:nums[3] + 1]
        if line.startswith('turn on'):
            grid[area] += 1
        elif line.startswith('turn off'):
            # restrict subtraction to non-negative values
            grid[area] = np.clip(np.subtract(grid[area], 1), 0, np.inf)
        elif line.startswith('toggle'):
            grid[area] += 2
    return np.sum(grid)


print(part_1(parsers.lines(loader.get())))  # 569999
print(part_2(parsers.lines(loader.get())))  # 17836115
