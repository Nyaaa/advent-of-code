import numpy as np

from tools import loader, parsers


def part1(data: list[str]) -> int:
    """
    >>> print(part1(parsers.lines('test.txt')))
    21
    """
    arr = np.genfromtxt(data, delimiter=1, dtype=str)
    beams = {tuple(np.argwhere(arr == 'S')[0])}
    seen = set()
    result = 0
    while beams:
        pos = beams.pop()
        next_row = pos[0] + 1
        below = (next_row, pos[1])
        if below[0] >= arr.shape[0] or below in seen:
            continue
        seen.add(pos)
        char = arr[below]
        if char == '.':
            beams.add(below)
        elif char == '^':
            beams.add((next_row, pos[1] - 1))
            beams.add((next_row, pos[1] + 1))
            result += 1
    return result


def part2(data: list[str]) -> int:
    """
    >>> print(part2(parsers.lines('test.txt')))
    40
    """
    arr = np.genfromtxt(data, delimiter=1, dtype=str)
    arr = np.flip(arr, axis=0)
    num_arr = np.zeros_like(arr, dtype=int)
    num_arr[0] = 1
    num_arr[np.where(arr == '^')] = -1
    for i, val in np.ndenumerate(num_arr):
        if val == 0:
            num_arr[i] = num_arr[i[0] - 1, i[1]]
        elif val == -1:
            num_arr[i] = num_arr[i[0] - 1, i[1] - 1] + num_arr[i[0] - 1, i[1] + 1]
    return num_arr[np.where(arr == 'S')][0]


print(part1(parsers.lines(loader.get())))  # 1490
print(part2(parsers.lines(loader.get())))  # 3806264447357
