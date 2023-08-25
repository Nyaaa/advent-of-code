from __future__ import annotations
from math import gcd
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

    def part_2(self):
        """
        >>> print(Trajectory(parsers.lines('test.txt')).part_2())
        2772

        >>> print(Trajectory(parsers.lines('test2.txt')).part_2())
        4686774924"""

        def lcm(a: int, b: int) -> int:
            return a * b // gcd(a, b)

        time = 0
        init_pos = [[x.location.x for x in self.moons],
                    [y.location.y for y in self.moons],
                    [z.location.z for z in self.moons]]
        steps = [0, 0, 0]
        while not all(steps):
            time += 1
            for moon1, moon2 in combinations(self.moons, 2):
                moon1.apply_gravity(moon2)
            for moon in self.moons:
                moon.apply_velocity()
            locs = [[j for j in i.location] for i in self.moons]
            vels = [[j for j in i.velocity] for i in self.moons]
            for i in range(3):
                if ([j[i] for j in locs] == init_pos[i]
                        and [j[i] for j in vels] == [0, 0, 0, 0]
                        and steps[i] == 0):
                    steps[i] = time
        return lcm(lcm(steps[0], steps[1]), steps[2])


print(Trajectory(parsers.lines(loader.get())).part_1(1000))  # 7928
print(Trajectory(parsers.lines(loader.get())).part_2())  # 518311327635164
