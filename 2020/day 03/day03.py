from math import prod
from numpy.typing import NDArray
from tools import parsers, loader
import numpy as np


def descend(grid: NDArray, step: tuple[int, int]) -> int:
    path = []
    location = (0, 0)
    while location[0] < grid.shape[0]:
        obj = grid[location]
        path.append(obj)
        location = (location[0] + step[0], location[1] + step[1])
    return sum(path[1:])


def start(data: list, part2: bool):
    """test part 1:
    >>> print(start(parsers.lines('test.txt'), False))
    7

    test part 2:
    >>> print(start(parsers.lines('test.txt'), True))
    336
    """
    data = np.asarray([list(i.replace('.', '0').replace('#', '1')) for i in data], dtype=int)
    data = np.hstack([data] * data.shape[0] * 4)
    steps = [(1, 1), (1, 3), (1, 5), (1, 7), (2, 1)]
    results = [descend(data, step) for step in steps]
    return results[1] if not part2 else prod(results)


print(start(parsers.lines(loader.get()), False))  # 282
print(start(parsers.lines(loader.get()), True))  # 958815792
