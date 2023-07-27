from __future__ import annotations
import re
from dataclasses import dataclass
from itertools import combinations

from tools import parsers, loader


@dataclass(frozen=True)
class Point:
    x: int
    y: int
    z: int

    def __add__(self, other: Point):
        return Point(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Point):
        return Point(self.x - other.x, self.y - other.y, self.z - other.z)

    def __iter__(self):
        yield from (self.x, self.y, self.z)

    def compare(self, other: Point):
        vals = []
        for i, j in zip(self, other):
            if i > j:
                vals.append(-1)
            elif i < j:
                vals.append(1)
            else:
                vals.append(0)
        return Point(*vals)


@dataclass
class Moon:
    location: Point
    velocity: Point

    def apply_gravity(self, other: Moon):
        diff = self.location.compare(other.location)
        self.velocity += diff
        other.velocity -= diff

    def apply_velocity(self):
        self.location += self.velocity

    def calculate_energy(self):
        return sum(abs(i) for i in self.location) * sum(abs(i) for i in self.velocity)


class Trajectory:
    def __init__(self, data: list):
        self.moons = []
        for line in data:
            coords = Point(*(int(i) for i in re.findall(r'-?\d+', line)))
            self.moons.append(Moon(coords, Point(0, 0, 0)))

    def part_1(self, steps: int):
        """
        >>> print(Trajectory(parsers.lines('test.txt')).part_1(10))
        179

        >>> print(Trajectory(parsers.lines('test2.txt')).part_1(100))
        1940
        """
        time = 0
        while time < steps:
            time += 1
            for moon1, moon2 in combinations(self.moons, 2):
                moon1.apply_gravity(moon2)
            for moon in self.moons:
                moon.apply_velocity()
        return sum(i.calculate_energy() for i in self.moons)


print(Trajectory(parsers.lines(loader.get())).part_1(1000))  # 7928

