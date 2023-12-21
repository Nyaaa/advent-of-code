from collections import deque

import numpy as np

from tools import common, loader, parsers


def garden(data: list[str], cutoff: int) -> int:
    """
    >>> print(garden(parsers.lines('test.txt'), cutoff=6))
    16"""
    arr = np.asarray([list(line) for line in data])
    start = tuple(np.argwhere(arr == 'S')[0])
    queue = deque([(0, start)])
    seen = set()
    part1 = 0
    while queue:
        distance, tile = queue.popleft()
        if tile in seen or distance > cutoff:
            continue
        seen.add(tile)
        if distance % 2 == 0:
            part1 += 1
        for i, val in common.get_adjacent(arr, tile):
            if i not in seen and val != '#':
                queue.append((distance + 1, i))

    return part1


print(garden(parsers.lines(loader.get()), cutoff=64))  # 3773
