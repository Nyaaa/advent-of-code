from pathlib import Path

import numpy as np

from tools import loader


def forest(data: str | Path) -> tuple[int, int]:
    """
    >>> print(forest('test.txt'))
    (21, 8)"""
    arr = np.genfromtxt(data, delimiter=1, dtype=int)
    part1 = part2 = 0

    for loc, val in np.ndenumerate(arr):
        left = arr[loc[0], :loc[1]][::-1]
        right = arr[loc[0], loc[1]+1:]
        top = arr[:loc[0], loc[1]][::-1]
        bottom = arr[loc[0]+1:, loc[1]]

        part1 += (np.all(val > left) or np.all(val > right) or
                  np.all(val > top) or np.all(val > bottom))

        score = 1
        for line in left, right, top, bottom:
            count = 0
            for tree in line:
                count += 1
                if tree >= val:
                    break
            score *= count
        part2 = max(part2, score)

    return part1, part2


print(forest(loader.get()))  # 1543, 595080
