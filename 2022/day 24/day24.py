from tools import parsers, loader
from collections import deque
from math import lcm
from typing import NamedTuple
import numpy as np


class Point(NamedTuple):
    row: int
    col: int

    def __add__(self, other):
        return Point(row=self.row + other.row, col=self.col + other.col)


DIRECTIONS = {'>': Point(0, 1), '<': Point(0, -1), '^': Point(-1, 0), 'v': Point(1, 0)}
ADJACENT = (Point(0, 1), Point(0, -1), Point(-1, 0), Point(1, 0), Point(0, 0))


class Blizzard:
    def __init__(self, data):
        self.data = np.asarray([list(line) for line in data])
        self.max_rows = len(self.data)
        self.max_cols = len(self.data[0])
        self.start = Point(row=0, col=1)
        self.end = Point(row=self.max_rows - 1, col=self.max_cols - 2)
        self.map_num = lcm(self.max_rows - 2, self.max_cols - 2)
        self.map_list = self.precalculate_maps()

    def precalculate_maps(self) -> list[set[Point]]:
        map_list = []
        wind_map = [(Point(r, c), d) for d in DIRECTIONS.keys() for r, c in list(zip(*np.where(self.data == d)))]
        for _ in range(self.map_num + 1):
            wind_map = self.generate_map(wind_map)
            map_list.append(set(point for point, d in wind_map))
        return map_list

    def generate_map(self, wind_map: list[tuple[Point, str]]) -> list[tuple[Point, str]]:
        _map = []
        for point, d in wind_map:
            row, col = point + DIRECTIONS[d]
            if row == self.max_rows - 1:
                row = 1
            elif row == 0:
                row = self.max_rows - 2

            if col == self.max_cols - 1:
                col = 1
            elif col == 0:
                col = self.max_cols - 2

            _map.append((Point(row, col), d))
        return _map

    def adjacent(self, point: Point, map_num: int) -> list[Point]:
        points = [point + adj for adj in ADJACENT]
        return [point for point in points if
                (1 <= point.row < self.max_rows - 1 and 1 <= point.col < self.max_cols - 1
                 or point in (self.start, self.end))
                and point not in self.map_list[map_num]]

    def search(self, start: Point, end: Point, time: int) -> int:
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

    def part_1(self):
        """
        >>> print(Blizzard(parsers.lines('test.txt')).part_1())
        18"""
        return self.search(self.start, self.end, 0)

    def part_2(self):
        """
        >>> print(Blizzard(parsers.lines('test.txt')).part_2())
        54"""
        time = self.search(self.start, self.end, 0)
        time += self.search(self.end, self.start, time)
        time += self.search(self.start, self.end, time)
        return time


print(Blizzard(parsers.lines(loader.get())).part_1())  # 245
print(Blizzard(parsers.lines(loader.get())).part_2())  # 798
