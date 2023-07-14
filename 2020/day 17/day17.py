from numpy.typing import NDArray
from tools import parsers, loader
import numpy as np


def cycle(arr: NDArray) -> NDArray:
    result = None
    for _ in range(6):
        arr = np.pad(arr, pad_width=1, mode='constant', constant_values=0)
        result = arr.copy()
        for (x, y, z), value in np.ndenumerate(arr):
            neighbours = arr[max(0, x - 1):x + 2, max(0, y - 1):y + 2, max(0, z - 1):z + 2]
            active = np.count_nonzero(neighbours) - value
            result[x, y, z] = 1 if active == 3 or (active == 2 and value == 1) else 0
        arr = result
    return result


def activate(data: list) -> int:
    """
    >>> print(activate(parsers.lines('test.txt')))
    112
    """
    arr = np.array([list(1 if i == '#' else 0 for i in row) for row in data])
    arr.shape = (1, arr.shape[0], arr.shape[1])
    return np.count_nonzero(cycle(arr))


print(activate(parsers.lines(loader.get())))  # 315
