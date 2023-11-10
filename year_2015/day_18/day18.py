import numpy as np

from tools import common, loader, parsers


def animation(data: list[str], steps: int, part2: bool) -> int:
    """
    >>> print(animation(parsers.lines('test.txt'), steps=4, part2=False))
    4
    >>> print(animation(parsers.lines('test.txt'), steps=5, part2=True))
    17"""
    arr = np.array([[1 if i == '#' else 0 for i in row] for row in data],
                   dtype=np.dtype('u1'))
    corners = np.ix_((0, -1), (0, -1))
    if part2:
        arr[corners] = 1
    step = 0
    while step != steps:
        arr_copy = np.zeros_like(arr)
        for i, val in np.ndenumerate(arr):
            adj = sum(j for i, j in common.get_adjacent(arr, i, with_corners=True))
            match val, adj:
                case 1, 2 | 3:
                    arr_copy[i] = 1
                case 0, 3:
                    arr_copy[i] = 1
                case _:
                    arr_copy[i] = 0
        arr = arr_copy
        if part2:
            arr[corners] = 1
        step += 1
    return np.count_nonzero(arr)


print(animation(parsers.lines(loader.get()), steps=100, part2=False))  # 821
print(animation(parsers.lines(loader.get()), steps=100, part2=True))  # 886
