import numpy as np
from numpy.typing import NDArray

from tools import loader, parsers


def tilt(arr: NDArray) -> NDArray:
    new_arr = []
    for i in range(arr.shape[1]):
        col = arr[:, i]
        divided = np.split(col, np.where(col == 0)[0])
        new_arr.append(np.hstack([np.sort(chunk) for chunk in divided]))
    return np.vstack(new_arr).T


def calculate_load(arr: NDArray) -> int:
    value = 0
    for i in range(arr.shape[1]):
        col = arr[::-1, i]
        stones = np.where(col == 1)[0]
        value += np.sum(stones) + len(stones)
    return value


def platform(data: list[str]) -> int:
    """
    >>> print(platform(parsers.lines('test.txt')))
    136"""
    arr = np.array([list(i) for i in data], dtype=str)
    arr[arr == 'O'] = 1
    arr[arr == '#'] = 0
    arr[arr == '.'] = 2
    arr = arr.astype(np.dtype('u1'))
    tilted = tilt(arr)
    return calculate_load(tilted)


print(platform(parsers.lines(loader.get())))  # 110274
