import numpy as np

from tools import loader, parsers


def part_1(data: list[str]) -> int:
    valid = 0
    for line in data:
        a, b, c = (int(i) for i in line.split())
        if a + b > c and a + c > b and b + c > a:
            valid += 1
    return valid


def part_2(data: list[str]) -> int:
    arr = np.genfromtxt(data, dtype=int)
    valid = 0
    arr = np.reshape(arr, (arr.shape[1], arr.shape[0]), order='F').transpose()
    for a, b, c in arr:
        if a + b > c and a + c > b and b + c > a:
            valid += 1
    return valid


print(part_1(parsers.lines(loader.get())))  # 1032
print(part_2(parsers.lines(loader.get())))  # 1838
