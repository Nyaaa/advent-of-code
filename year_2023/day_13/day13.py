import numpy as np
from numpy.typing import NDArray

from tools import loader, parsers


def mirrors(data: list[list[str]], part2: bool) -> int:
    """
    >>> print(mirrors(parsers.blocks('test.txt'), part2=False))
    405
    >>> print(mirrors(parsers.blocks('test.txt'), part2=True))
    400"""
    def find_split(arr: NDArray) -> int:
        size = arr.shape[0]
        for row in range(1, size):
            split_size = min(row, size - row)
            top, bottom = np.vsplit(arr[row - split_size: row + split_size], 2)
            if np.sum(top != np.flipud(bottom)) == int(part2):
                return row
        return 0

    result = 0
    for block in data:
        pattern = np.asarray([[j == '#' for j in i] for i in block], dtype=bool)
        if horizontal := find_split(pattern):
            result += horizontal * 100
        else:
            result += find_split(pattern.T)
    return result


print(mirrors(parsers.blocks(loader.get()), part2=False))  # 37975
print(mirrors(parsers.blocks(loader.get()), part2=True))  # 32497
