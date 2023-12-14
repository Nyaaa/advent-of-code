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


def platform(data: list[str]) -> tuple[int, int]:
    """
    >>> print(platform(parsers.lines('test.txt')))
    (136, 64)"""
    arr = np.array([list(i) for i in data], dtype=str)
    arr[arr == 'O'] = 1
    arr[arr == '#'] = 0
    arr[arr == '.'] = 2
    arr = arr.astype(np.dtype('u1'))
    part1 = calculate_load(tilt(arr))

    seen = {}
    cycle = 0
    while cycle <= 1_000_000_000:
        for _ in range(4):
            arr = tilt(arr)
            arr = np.rot90(arr, -1)
        cycle += 1
        hashable = arr.tobytes()
        if hashable in seen and cycle < 100_000:
            cycle_len = cycle - seen[hashable]
            cycles_left = (1000000000 - cycle) % cycle_len
            cycle = 1000000000 - cycles_left + 1
        seen[hashable] = cycle
    return part1, calculate_load(arr)


print(platform(parsers.lines(loader.get())))  # 110274, 90982
