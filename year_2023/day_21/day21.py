from collections import deque
from collections.abc import Generator

import numpy as np
from numpy.polynomial.polynomial import Polynomial

from tools import loader, parsers


class Garden:
    def __init__(self, data: list[str]) -> None:
        self.arr = np.asarray([list(line) for line in data])
        self.start = tuple(np.argwhere(self.arr == 'S')[0])
        self.size = self.arr.shape[0]

    def part1(self, cutoff: int) -> int:
        """
        >>> print(Garden(parsers.lines('test.txt')).part1(cutoff=6))
        16
        >>> print(Garden(parsers.lines('test.txt')).part1(cutoff=50))
        1594"""
        queue = deque([(0, self.start)])
        seen = set()
        result = 0
        while queue:
            distance, tile = queue.popleft()
            if tile in seen or distance > cutoff:
                continue
            seen.add(tile)
            if distance % 2 == cutoff % 2:
                result += 1
            for i, val in self.get_inf_adjacent(tile):
                if i not in seen and val != '#':
                    queue.append((distance + 1, i))
        return result

    def get_inf_adjacent(self, location: tuple[int, int]
                         ) -> Generator[tuple[tuple[int, int]], str]:
        for i in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            x, y = location[0] + i[0], location[1] + i[1]
            yield (x, y), self.arr[x % self.size, y % self.size]

    def part2(self) -> int:
        x = [0, 1, 2]
        y = [self.part1(cutoff=65), self.part1(cutoff=65 + 131), self.part1(cutoff=65 + 2 * 131)]
        poly = Polynomial.fit(x, y, deg=2)
        return int(poly((26501365 - 65) // 131))


print(Garden(parsers.lines(loader.get())).part1(cutoff=64))  # 3773
print(Garden(parsers.lines(loader.get())).part2())  # 625628021226274
