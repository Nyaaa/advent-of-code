from pathlib import Path

import numpy as np
from numpy.polynomial.polynomial import Polynomial

from tools import loader


def oasis(data: str | Path) -> tuple[int, int]:
    """
    >>> print(oasis('test.txt'))
    (114, 2)"""
    arr = np.genfromtxt(data, dtype=int)
    length = arr.shape[1]
    polynoms = [Polynomial.fit(range(length), line, deg=length - 1) for line in arr]
    part1 = sum(round(p(length)) for p in polynoms)
    part2 = sum(round(p(-1)) for p in polynoms)
    return part1, part2


print(oasis(loader.get()))  # 1993300041, 1038
