from tools import parsers, loader
from typing import NamedTuple
from math import prod

TEST = """2199943210
3987894921
9856789892
8767896789
9899965678
"""


class Point(NamedTuple):
    row: int
    col: int
    height: int


class Floor:
    def __init__(self, data):
        self.grid = [[int(i) for i in row] for row in data]
        self.cols = len(self.grid[0])
        self.rows = len(self.grid)

    def neighbours(self, row: int, col: int):
        for i, j in (-1, 0), (1, 0), (0, -1), (0, 1):
            r, c = row + i, col + j
            if r >= 0 and c >= 0:
                try:
                    yield Point(r, c, self.grid[r][c])
                except IndexError:
                    continue

    def get_lowest(self):
        for row in range(self.rows):
            for col in range(self.cols):
                point = self.grid[row][col]
                if all([point < i.height for i in list(self.neighbours(row, col))]):
                    yield Point(row, col, point)

    def get_pool(self, points: set) -> set[Point]:
        for point in set(points):
            nb = set([i for i in self.neighbours(point.row, point.col) if i.height < 9 and i not in points])
            if nb:
                points.update(nb)
                points.update(self.get_pool(points))
        return points

    def part_1(self):
        """
        >>> print(Floor(parsers.inline_test(TEST)).part_1())
        15"""
        return sum([i.height + 1 for i in self.get_lowest()])

    def part_2(self):
        """
        >>> print(Floor(parsers.inline_test(TEST)).part_2())
        1134"""
        pools = [len(self.get_pool({point})) for point in self.get_lowest()]
        return prod(sorted(pools, reverse=True)[0:3])


print(Floor(parsers.lines(loader.get())).part_1())  # 631
print(Floor(parsers.lines(loader.get())).part_2())  # 821560
