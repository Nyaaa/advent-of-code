from __future__ import annotations
from dataclasses import dataclass
from typing import NamedTuple
from tools import parsers, loader
import re


class Side(NamedTuple):
    min: int
    max: int

    def __contains__(self, item: Side) -> bool:
        return self.min <= item.min and self.max >= item.max

    def intersects(self, other: Side) -> bool:
        return self.min <= other.max and self.max >= other.min

    def split_side(self, other: Side) -> tuple[Side, Side]:
        if self.min < other.min <= self.max:
            return Side(self.min, other.min - 1), Side(other.min, self.max)
        elif self.min <= other.max < self.max:
            return Side(other.max + 1, self.max), Side(self.min, other.max)
        raise IndexError


@dataclass
class Cube:
    x: Side
    y: Side
    z: Side
    op: bool = True

    def __contains__(self, item: Cube) -> bool:
        return item.x in self.x and item.y in self.y and item.z in self.z

    def intersects(self, other: Cube) -> bool:
        return self.x.intersects(other.x) and self.y.intersects(other.y) and self.z.intersects(other.z)

    def size(self) -> int:
        return (self.x.max - self.x.min + 1) * (self.y.max - self.y.min + 1) * (self.z.max - self.z.min + 1)

    def split(self, other: Cube) -> list[Cube]:
        new = []
        for i, j, side in (self.x, other.x, 'x'), (self.y, other.y, 'y'), (self.z, other.z, 'z'):
            try:
                new_self, new_other = i.split_side(j)
            except IndexError:
                continue
            else:
                if side == 'x':
                    new.append(Cube(new_other, self.y, self.z))
                    self.x = new_self
                if side == 'y':
                    new.append(Cube(self.x, new_other, self.z))
                    self.y = new_self
                if side == 'z':
                    new.append(Cube(self.x, self.y, new_other))
                    self.z = new_self
        return new


def generate_cuboids(data: list[str], limit: bool) -> list[Cube]:
    commands = []
    for line in data:
        op = True if line.split(' ')[0] == 'on' else False
        vals = [int(i) for i in re.findall(r'-?\d+', line)]
        if limit and (min(vals) < -50 or max(vals) > 50):
            continue
        commands.append(Cube(op=op,
                             x=Side(vals[0], vals[1]),
                             y=Side(vals[2], vals[3]),
                             z=Side(vals[4], vals[5])))
    return commands


def start(data: list[str], limit: bool):
    """ test part 1:
    >>> print(start(parsers.lines('test.txt'), limit=True))
    474140

    test part 2:
    >>> print(start(parsers.lines('test.txt'), limit=False))
    2758514936282235"""
    commands = generate_cuboids(data, limit)
    result = []
    for command in commands:
        while True:
            for cube in result:
                if cube in command:
                    result.remove(cube)
                    break
                if command.intersects(cube):
                    result.extend(cube.split(command))
            else:
                break
        if command.op:
            result.append(command)
    return sum(i.size() for i in result)


print(start(parsers.lines(loader.get()), limit=True))  # 611176
print(start(parsers.lines(loader.get()), limit=False))  # 1201259791805392
