from tools import parsers, loader
import numpy as np
from collections import deque


class Octopi:
    def __init__(self, data):
        data = [list(i) for i in data]
        self.grid = np.asarray(data, dtype=int)

    def get_adjacent(self, position: tuple[int, int]) -> list[tuple[int, int]]:
        adj = []

        for row in range(-1, 2):
            for col in range(-1, 2):
                max_rows, max_cols = self.grid.shape
                adj_row = position[0] + row
                adj_col = position[1] + col

                if (0 <= adj_row < max_cols) and (0 <= adj_col < max_rows) and (row, col) != (0, 0):
                    adj.append((adj_row, adj_col))

        return adj

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
            flashed: deque[tuple[int, int]] = deque(zip(fl[0], fl[1]))  # row, col
            done = []
            while flashed:
                i = flashed.popleft()
                done.append(i)
                adj = self.get_adjacent(i)
                for j in adj:
                    self.grid[j] += 1
                    if self.grid[j] > 9 and j not in flashed and j not in done:
                        flashed.append(j)
            flashes += len(done)

            self.grid[self.grid >= 10] = 0
            if not part2 and step == 100:
                return flashes
            elif part2 and not self.grid.any():
                return step


print(Octopi(parsers.lines(loader.get())).cycle())  # 1700
print(Octopi(parsers.lines(loader.get())).cycle(part2=True))  # 273
