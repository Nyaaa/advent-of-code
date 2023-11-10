import numpy as np
from numba import njit

from tools import common, loader, parsers


@njit
def animation(data: tuple[str], steps: int, part2: bool) -> int:
    """
    >>> print(animation(tuple(parsers.lines('test.txt')), steps=4, part2=False))
    4
    >>> print(animation(tuple(parsers.lines('test.txt')), steps=5, part2=True))
    17"""
    arr = np.array([[1 if i == '#' else 0 for i in row] for row in data],
                   dtype=np.dtype('u1'))
    m, n = arr.shape
    if part2:
        arr[::m - 1, ::n - 1] = 1
    step = 0
    while step != steps:
        arr_copy = np.zeros_like(arr)
        for i, val in np.ndenumerate(arr):
            adj = sum([j for i, j in common.get_adjacent(arr, i, with_corners=True)])
            if (val == 1 and adj in (2, 3)) or (val == 0 and adj == 3):
                arr_copy[i] = 1
            else:
                arr_copy[i] = 0
        arr = arr_copy
        if part2:
            arr[::m - 1, ::n - 1] = 1
        step += 1
    return np.count_nonzero(arr)


print(animation(tuple(parsers.lines(loader.get())), steps=100, part2=False))  # 821
print(animation(tuple(parsers.lines(loader.get())), steps=100, part2=True))  # 886
