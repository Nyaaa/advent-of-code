import numpy as np

from tools import common, loader, parsers


def rolls(data: list[str], part2: bool) -> int:
    """
    >>> print(rolls(parsers.lines('test.txt'), False))
    13
    >>> print(rolls(parsers.lines('test.txt'), True))
    43
    """
    result = 0
    arr = np.array([[1 if i == '@' else 0 for i in row] for row in data], dtype=np.dtype('u1'))
    while True:
        moved = 0
        for i, val in np.ndenumerate(arr):
            adj = sum(j for _, j in common.get_adjacent(arr, i, with_corners=True))
            if val == 1 and adj < 4:
                result += 1
                if part2:
                    moved += 1
                    arr[i] = 0
        if moved == 0:
            return result


print(rolls(parsers.lines(loader.get()), False))  # 1467
print(rolls(parsers.lines(loader.get()), True))  # 8484
