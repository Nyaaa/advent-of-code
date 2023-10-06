import numpy as np
from numba import njit
from numpy.typing import NDArray

from tools import loader, parsers


@njit(fastmath=True)
def fft(arr: NDArray, pattern: NDArray) -> NDArray:
    arr_len = arr.shape[0]
    for _ in range(100):
        new = arr.copy()
        for i in range(arr_len):
            p = np.repeat(pattern, i + 1)[1:arr_len + 1]
            new[i] = np.abs(np.sum(arr * p)) % 10
        arr = new
    return arr


def part_1(data: str) -> str:
    """
    >>> print(part_1('80871224585914546619083218645595'))
    24176176"""
    pattern = np.asarray([0, 1, 0, -1], dtype=np.dtype('i1'))
    arr = np.fromiter(data, dtype=np.dtype('i1'))
    reps = int(np.ceil(arr.shape[0] / 4)) + 1
    arr = fft(arr, np.tile(pattern, reps))
    return ''.join(str(i) for i in arr[:8])


def part_2(data: str) -> str:
    """
    >>> print(part_2('03036732577212944063491565474664'))
    84462026"""
    data *= 10000
    offset = int(data[0:7])
    arr = np.fromiter(data[offset:], dtype=np.dtype('i1'))
    for _ in range(100):
        new = np.cumsum(arr[::-1]) % 10
        arr = new[::-1]
    return ''.join(str(i) for i in arr[:8])


print(part_1(parsers.string(loader.get())))  # 94960436
print(part_2(parsers.string(loader.get())))  # 57762756
