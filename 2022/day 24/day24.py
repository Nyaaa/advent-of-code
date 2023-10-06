from collections import deque
from math import lcm

import numpy as np

from tools import loader, parsers

DIRECTIONS = {'>': 1j, '<': -1j, '^': -1, 'v': 1}


class Blizzard:
    def __init__(self, data: list[str]) -> None:
        self.data = np.asarray([list(line) for line in data])
        self.max_rows, self.max_cols = self.data.shape
        self.start = 1j
        self.end = complex(self.max_rows - 1, self.max_cols - 2)
        self.map_num = lcm(self.max_rows - 2, self.max_cols - 2)
        self.map_list = self.precalculate_maps()

    def precalculate_maps(self) -> list[set[complex]]:
        map_list = []
        wind_map = [(complex(r, c), d) for d in DIRECTIONS
                    for r, c in list(zip(*np.where(self.data == d)))]
        for _ in range(self.map_num + 1):
            wind_map = self.generate_map(wind_map)
            map_list.append({point for point, d in wind_map})
        return map_list

    def generate_map(self, wind_map: list[tuple[complex, str]]) -> list[tuple[complex, str]]:
        _map = []
        for point, d in wind_map:
            i = point + DIRECTIONS[d]
            row, col = i.real, i.imag
            if row == self.max_rows - 1:
                row = 1
            elif row == 0:
                row = self.max_rows - 2

            if col == self.max_cols - 1:
                col = 1
            elif col == 0:
                col = self.max_cols - 2

            _map.append((complex(row, col), d))
        return _map

    def adjacent(self, point: complex, map_num: int) -> list[complex]:
        points = [point + adj for adj in (1j, -1j, -1, 1, 0)]
        return [point for point in points if
                (1 <= point.real < self.max_rows - 1 and 1 <= point.imag < self.max_cols - 1
                 or point in (self.start, self.end))
                and point not in self.map_list[map_num]]

    def search(self, start: complex, end: complex, time: int) -> int:
        queue = deque([(start, time)])
        done = set()
        _time = 0
        while start != end:
            start, _time = queue.popleft()
            _map_num = _time % self.map_num
            if (start, _map_num) not in done:
                done.add((start, _map_num))
                queue += [(next_point, _time + 1) for next_point in self.adjacent(start, _map_num)]
        return _time - time

    def part_1(self) -> int:
        """
        >>> print(Blizzard(parsers.lines('test.txt')).part_1())
        18"""
        return self.search(self.start, self.end, 0)

    def part_2(self) -> int:
        """
        >>> print(Blizzard(parsers.lines('test.txt')).part_2())
        54"""
        time = self.search(self.start, self.end, 0)
        time += self.search(self.end, self.start, time)
        time += self.search(self.start, self.end, time)
        return time


print(Blizzard(parsers.lines(loader.get())).part_1())  # 245
print(Blizzard(parsers.lines(loader.get())).part_2())  # 798
