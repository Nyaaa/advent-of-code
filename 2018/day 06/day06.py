from tools import parsers, loader
import numpy as np


class Area:
    def __init__(self, data: list[str]) -> None:
        self.points = {(int(row), int(col)): 0 for col, row in (line.split(', ') for line in data)}
        self.max_col = max(self.points, key=lambda x: x[1])[1]
        self.max_row = max(self.points, key=lambda x: x[0])[0]
        self.grid = np.zeros((self.max_row + 2, self.max_col + 2), dtype=int)
        for p in self.points:
            self.grid[p] = 1
        self.infinite = set()

    @staticmethod
    def manhattan_distance(point: tuple[int, ...], other: tuple[int, ...]) -> int:
        return abs(point[0] - other[0]) + abs(point[1] - other[1])

    def start(self, max_distance: int) -> tuple[int, int]:
        """
        >>> print(Area(parsers.lines('test.txt')).start(32))
        (17, 16)"""
        part2 = 0
        for point in np.ndindex(self.grid.shape):
            distances = [(self.manhattan_distance(point, i), i) for i in self.points]
            if sum(i[0] for i in distances) < max_distance:
                part2 += 1
            closest_point = min(distances, key=lambda x: x[0])[1]
            self.points[closest_point] += 1
            if point[0] == 0 or point[0] == self.max_row or point[1] == 0 or point[1] == self.max_col:
                self.infinite.add(closest_point)
        part1 = max(v for k, v in self.points.items() if k not in self.infinite)
        return part1, part2


print(Area(parsers.lines(loader.get())).start(10000))  # 5532, 36216
