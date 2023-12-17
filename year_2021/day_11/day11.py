from collections import deque

import numpy as np

from tools import common, loader, parsers


class Octopi:
    def __init__(self, data: list[str]) -> None:
        data = [list(i) for i in data]
        self.grid = np.asarray(data, dtype=int)

    def cycle(self, part2: bool = False) -> int:
        """
        >>> print(Octopi(parsers.lines('test.txt')).cycle())
        1656

        >>> print(Octopi(parsers.lines('test.txt')).cycle(part2=True))
        195
        """
        step = 0
        flashes = 0
        while True:
            step += 1
            self.grid += 1
            fl = np.where(self.grid == 10)
            flashed: deque[tuple[int, int]] = deque(zip(fl[0], fl[1], strict=True))  # row, col
            done = []
            while flashed:
                i = flashed.popleft()
                done.append(i)
                for index, _value in common.get_adjacent(self.grid, i, with_corners=True):
                    self.grid[index] += 1
                    if self.grid[index] > 9 and index not in flashed and index not in done:
                        flashed.append(index)
            flashes += len(done)

            self.grid[self.grid >= 10] = 0
            if not part2 and step == 100:
                return flashes
            if part2 and not self.grid.any():
                return step


print(Octopi(parsers.lines(loader.get())).cycle())  # 1700
print(Octopi(parsers.lines(loader.get())).cycle(part2=True))  # 273
