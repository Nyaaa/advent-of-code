import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from tools import parsers, loader


def part_1(data: list[str]) -> int:
    valid = 0
    for line in data:
        a, b, c = (int(i) for i in line.split())
        if a + b > c and a + c > b and b + c > a:
            valid += 1
    return valid


def part_2(data: list[str]) -> int:
    arr = np.asarray([line.split() for line in data], dtype=int)
    valid = 0
    arr = np.reshape(arr, (arr.shape[1], arr.shape[0]), order='F').transpose()
    for a, b, c in arr:
        if a + b > c and a + c > b and b + c > a:
            valid += 1
    return valid


print(part_1(parsers.lines(loader.get())))  # 1032
print(part_2(parsers.lines(loader.get())))  # 1838
