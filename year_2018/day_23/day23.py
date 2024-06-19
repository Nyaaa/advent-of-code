from __future__ import annotations

import re
from collections.abc import Generator
from dataclasses import dataclass
from functools import cached_property
from itertools import product, starmap
from queue import PriorityQueue

from tools import loader, parsers


@dataclass(frozen=True)
class Point:
    x: int
    y: int
    z: int

    def distance(self, other: Point) -> int:
        return abs(self.x - other.x) + abs(self.y - other.y) + abs(self.z - other.z)

    def __add__(self, other: int | Point) -> Point:
        if isinstance(other, int):
            return Point(self.x + other, self.y + other, self.z + other)
        return Point(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: int) -> Point:
        return Point(self.x - other, self.y - other, self.z - other)

    def __mul__(self, value: int) -> Point:
        return Point(self.x * value, self.y * value, self.z * value)


@dataclass(frozen=True)
class Nanobot(Point):
    r: int

    def is_in_range(self, other: Nanobot) -> bool:
        return self.distance(other) <= self.r

    def max_axis(self) -> int:
        return max(abs(i) + self.r for i in (self.x, self.y, self.z))


@dataclass(frozen=True)
class BoundingBox:
    corner_a: Point
    corner_b: Point
    bots: list[Nanobot]
    size: int

    @cached_property
    def distance(self) -> int:
        return self.corner_a.distance(Point(0, 0, 0))

    @property
    def priority(self) -> tuple[int, int, int]:
        return -self.bots_in_range, -self.size, self.distance

    def __lt__(self, other: BoundingBox) -> bool:
        return self.priority < other.priority

    @cached_property
    def bots_in_range(self) -> int:
        return sum(
            (bot.distance(self.corner_a)
             + bot.distance(self.corner_b - 1)
             - self.corner_a.distance(self.corner_b - 1)
             ) // 2 <= bot.r
            for bot in self.bots)

    def generate_new(self) -> Generator[BoundingBox]:
        new_size = self.size // 2
        for octant in starmap(Point, product((0, 1), repeat=3)):
            corner_a = self.corner_a + octant * new_size
            yield BoundingBox(corner_a=corner_a,
                              corner_b=corner_a + new_size,
                              bots=self.bots,
                              size=new_size)


class Teleport:
    def __init__(self, data: list[str]) -> None:
        self.bots = [Nanobot(*map(int, re.findall(r'-?\d+', line))) for line in data]

    def part_1(self) -> int:
        """
        >>> print(Teleport(parsers.lines('test.txt')).part_1())
        7"""
        largest = max(self.bots, key=lambda i: i.r)
        return sum(largest.is_in_range(i) for i in self.bots)

    def part_2(self) -> int:
        """
        >>> print(Teleport(parsers.lines('test2.txt')).part_2())
        36"""
        size = 1
        while size <= max(i.max_axis() for i in self.bots):
            size *= 2
        curr_box = BoundingBox(corner_a=Point(-size, -size, -size),
                               corner_b=Point(size, size, size),
                               bots=self.bots,
                               size=2 * size)
        queue = PriorityQueue()
        queue.put(curr_box)
        while curr_box.size != 1:
            curr_box = queue.get()
            for new_box in curr_box.generate_new():
                queue.put(new_box)
        return curr_box.distance


print(Teleport(parsers.lines(loader.get())).part_1())  # 433
print(Teleport(parsers.lines(loader.get())).part_2())  # 107272899
