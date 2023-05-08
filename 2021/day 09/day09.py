from typing import Iterator

from tools import parsers, loader, common
from math import prod
import numpy as np

TEST = """2199943210
3987894921
9856789892
8767896789
9899965678
"""


class Floor:
    def __init__(self, data):
        self.grid = np.asarray([[int(i) for i in row] for row in data])
        self.rows, self.cols = self.grid.shape

    def get_lowest(self) -> Iterator[tuple[tuple[int, int], int]]:
        for index, point in np.ndenumerate(self.grid):
            if all([point < val for i, val in common.get_adjacent(self.grid, index)]):
                yield index, point

    def get_pool(self, points: set) -> set[tuple]:
        for point in set(points):
            nb = set([i for i in common.get_adjacent(self.grid, point[0]) if i[1] < 9 and i not in points])
            if nb:
                points.update(nb)
                points.update(self.get_pool(points))
        return points

    def part_1(self):
        """
        >>> print(Floor(parsers.inline_test(TEST)).part_1())
        15"""
        return sum([i[1] + 1 for i in self.get_lowest()])

    def part_2(self):
        """
        >>> print(Floor(parsers.inline_test(TEST)).part_2())
        1134"""
        pools = [len(self.get_pool({point})) for point in self.get_lowest()]
        return prod(sorted(pools, reverse=True)[0:3])


print(Floor(parsers.lines(loader.get())).part_1())  # 631
print(Floor(parsers.lines(loader.get())).part_2())  # 821560
