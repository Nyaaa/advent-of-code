from itertools import combinations

import numpy as np
from more_itertools import minmax

from tools import loader, parsers


def scan(data: list[str], multiplier: int) -> int:
    """
    >>> print(scan(parsers.lines('test.txt'), multiplier=2))
    374
    >>> print(scan(parsers.lines('test.txt'), multiplier=100))
    8410"""
    arr = np.asarray([[1 if i == '#' else 0 for i in row] for row in data],
                     dtype=np.dtype('u1'))
    cols = set(np.where(np.all(arr == 0, axis=0))[0])
    rows = set(np.where(np.all(arr == 0, axis=1))[0])
    galaxies = np.argwhere(arr != 0)
    distance = 0
    for a, b in combinations(galaxies, 2):
        empty_rows = sum(i in rows for i in range(*minmax(a[0], b[0])))
        empty_cols = sum(i in cols for i in range(*minmax(a[1], b[1])))
        distance += abs(a[0] - b[0]) + empty_rows * (multiplier - 1)
        distance += abs(a[1] - b[1]) + empty_cols * (multiplier - 1)
    return distance


print(scan(parsers.lines(loader.get()), multiplier=2))  # 9623138
print(scan(parsers.lines(loader.get()), multiplier=1_000_000))  # 726820169514
