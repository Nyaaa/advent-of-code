from numbers import Integral

import numpy as np
from numba import njit

from tools import loader, parsers


@njit
def calculate_presents(target: int) -> tuple[Integral, Integral]:
    upper_bound = 1_000_000
    houses = np.zeros((2, upper_bound), dtype=np.int32)
    for i in range(1, upper_bound):
        houses[0][i::i] += 10 * i
        houses[1][i:i * 50 + 1:i] += 11 * i
    part1 = np.flatnonzero(houses[0] >= target)[0]
    part2 = np.flatnonzero(houses[1] >= target)[0]
    return part1, part2


print(calculate_presents(int(parsers.string(loader.get()))))  # 831600, 884520
