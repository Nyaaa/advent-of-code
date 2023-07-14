from numpy.typing import NDArray
from tools import parsers, loader
import numpy as np


def cycle(arr: NDArray) -> NDArray:
    result = None
    for _ in range(6):
        arr = np.pad(arr, pad_width=1, mode='constant', constant_values=0)
        result = arr.copy()
        for index, value in np.ndenumerate(arr):
            neighbours = arr[*((slice(max(0, i - 1), i + 2)) for i in index)]
            active = np.count_nonzero(neighbours) - value
            result[index] = 1 if active == 3 or (active == 2 and value == 1) else 0
        arr = result
    return result


def activate(data: list, part2: bool = False) -> int:
    """
    >>> print(activate(parsers.lines('test.txt')))
    112

    >>> print(activate(parsers.lines('test.txt'), part2=True))
    848
    """
    arr = np.array([list(1 if i == '#' else 0 for i in row) for row in data])
    if not part2:
        arr.shape += (1, )
    else:
        arr.shape += (1, 1)
    return np.count_nonzero(cycle(arr))


print(activate(parsers.lines(loader.get())))  # 315
print(activate(parsers.lines(loader.get()), part2=True))  # 1520
