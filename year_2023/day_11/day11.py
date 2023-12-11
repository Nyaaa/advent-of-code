from itertools import combinations

import numpy as np

from tools import loader, parsers


def scan(data: list[str]) -> int:
    """
    >>> print(scan(parsers.lines('test.txt')))
    374"""
    arr = np.asarray([[1 if i == '#' else 0 for i in row] for row in data],
                     dtype=np.dtype('u1'))
    cols = np.argwhere(np.all(arr[..., :] == 0, axis=0))
    rows = np.argwhere(np.all(arr[..., :] == 0, axis=1))
    for c in cols[::-1]:
        arr = np.insert(arr, c[0], arr[:, c[0]], axis=1)
    for r in rows[::-1]:
        arr = np.insert(arr, r[0], arr[r[0]], axis=0)
    galaxies = np.argwhere(arr != 0)
    paths = []
    for a, b in combinations(galaxies, 2):
        paths.append(abs(a[0] - b[0]) + abs(a[1] - b[1]))

    return sum(paths)


print(scan(parsers.lines(loader.get())))  # 9623138
