from collections import Counter
from itertools import count

import numpy as np

from tools import common, loader, parsers


class Lumber:
    def __init__(self, data: list[str]) -> None:
        arr = np.asarray([list(i) for i in data])
        arr[arr == '#'] = 2  # lumberyard
        arr[arr == '|'] = 1  # tree
        arr[arr == '.'] = 0  # open
        self.forest = arr.astype(np.dtype('u1'))

    def cycle(self, n: int) -> int:
        for _ in range(n):
            new_forest = np.zeros_like(self.forest)
            for i, val in np.ndenumerate(self.forest):
                adj = [i[1] for i in common.get_adjacent(self.forest, i, with_corners=True)]
                counts = Counter(adj)
                if val == 0:
                    new_forest[i] = 1 if counts.get(1, 0) >= 3 else val
                if val == 1:
                    new_forest[i] = 2 if counts.get(2, 0) >= 3 else val
                if val == 2:
                    new_forest[i] = val if counts.get(2, 0) >= 1 and counts.get(1, 0) >= 1 else 0
            self.forest = new_forest
        return np.count_nonzero(self.forest == 1) * np.count_nonzero(self.forest == 2)

    def part_1(self) -> int:
        """
        >>> print(Lumber(parsers.lines('test.txt')).part_1())
        1147"""
        return self.cycle(10)

    def part_2(self) -> int:
        cycles = set()
        repetitions = {}
        for i in count(1):
            res = self.cycle(1)
            if res in cycles:
                if res in repetitions:
                    distance = i - repetitions[res]
                    if distance > 1:
                        remaining = (1_000_000_000 - i) % distance
                        return self.cycle(remaining)
                repetitions[res] = i
            cycles.add(res)
        raise ValueError('Solution not found.')


print(Lumber(parsers.lines(loader.get())).part_1())  # 483840
print(Lumber(parsers.lines(loader.get())).part_2())  # 219919
