from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from typing import NamedTuple
from tools import parsers, loader
import re


TEST = """p=< 3,0,0>, v=< 2,0,0>, a=<-1,0,0>
p=< 4,0,0>, v=< 0,0,0>, a=<-2,0,0>"""
TEST_2 = """p=<-6,0,0>, v=< 3,0,0>, a=< 0,0,0>
p=<-4,0,0>, v=< 2,0,0>, a=< 0,0,0>
p=<-2,0,0>, v=< 1,0,0>, a=< 0,0,0>
p=< 3,0,0>, v=<-1,0,0>, a=< 0,0,0>"""


class Point(NamedTuple):
    x: int
    y: int
    z: int

    def __add__(self, other: Point) -> Point:
        return Point(self.x + other.x, self.y + other.y, self.z + other.z)

    def __eq__(self, other: Point) -> bool:
        return self.x == other.x and self.y == other.y and self.z == other.z


@dataclass
class Particle:
    id: int
    position: Point
    velocity: Point
    acceleration: Point

    def step(self) -> None:
        self.velocity += self.acceleration
        self.position += self.velocity

    def distance(self) -> int:
        x, y, z = self.position
        return abs(0 - x) + abs(0 - y) + abs(0 - z)

    def __hash__(self) -> int:
        return hash(self.position)

    def __eq__(self, other: Particle) -> bool:
        if isinstance(other, Particle):
            return self.position == other.position
        return False


class Simulation:
    def __init__(self, data: list[str]) -> None:
        self.points = []
        for i, line in enumerate(data):
            nums = [int(i) for i in re.findall(r'-?\d+', line)]
            self.points.append(Particle(i, Point(*nums[0:3]), Point(*nums[3:6]), Point(*nums[6:9])))

    def part_1(self) -> int:
        """
        >>> print(Simulation(parsers.inline_test(TEST)).part_1())
        0"""
        prev_closest = None
        times_closest = 0
        while times_closest <= 500:  # may not be enough
            closest = min(self.points, key=lambda x: x.distance())
            if closest == prev_closest:
                times_closest += 1
            else:
                prev_closest = closest
                times_closest = 1
            for i in self.points:
                i.step()
        return prev_closest.id

    def part_2(self) -> int:
        """
        >>> print(Simulation(parsers.inline_test(TEST_2)).part_2())
        1"""
        min_len = float('inf')
        times = 0
        while times <= 500:
            for i, num in Counter(self.points).most_common():
                if num > 1:
                    self.points = [j for j in self.points if i.position != j.position]
            ln = len(self.points)
            if ln < min_len:
                min_len = ln
                times = 0
            elif ln == min_len:
                times += 1
            if ln <= 1:
                break

            for i in self.points:
                i.step()
        return min_len


print(Simulation(parsers.lines(loader.get())).part_1())  # 300
print(Simulation(parsers.lines(loader.get())).part_2())  # 502
