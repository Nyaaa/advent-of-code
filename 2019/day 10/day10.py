from __future__ import annotations
from collections import defaultdict, deque
from typing import NamedTuple
from tools import parsers, loader
import math
from itertools import permutations


class Asteroid(NamedTuple):
    row: int
    col: int

    def get_angle(self, other: Asteroid):
        result = math.atan2(other.col - self.col, self.row - other.row) * 180 / math.pi
        return 360 + result if result < 0 else result

    def distance(self, other: Asteroid):
        return abs(self.col - other.col) + abs(self.row - other.row)


class Space:
    def __init__(self, data: list[str]):
        self.asteroids = []
        for i, row in enumerate(data):
            for j, cell in enumerate(row):
                if cell == '#':
                    self.asteroids.append(Asteroid(i, j))
        self.laser = None

    def part_1(self):
        """
        >>> print(Space(parsers.lines('test.txt')).part_1())
        210"""
        vision = defaultdict(set)
        for start, stop in permutations(self.asteroids, 2):
            vision[start].add(start.get_angle(stop))
        self.laser, result = max(vision.items(), key=lambda x: len(x[1]))
        return len(result)

    def part_2(self):
        """
        >>> print(Space(parsers.lines('test.txt')).part_2())
        802"""
        self.part_1()
        self.asteroids.remove(self.laser)
        angles = [(self.laser.get_angle(end), end) for end in self.asteroids]
        sort = deque(sorted(angles, key=lambda x: (x[0], x[1].distance(self.laser))))
        evaporated = [sort.popleft()]

        while len(evaporated) < 200:
            if evaporated[-1][0] == sort[0][0]:
                sort.rotate(-1)
                continue
            evaporated.append(sort.popleft())
        return evaporated[-1][1][1] * 100 + evaporated[-1][1][0]


print(Space(parsers.lines(loader.get())).part_1())  # 221
print(Space(parsers.lines(loader.get())).part_2())  # 806
