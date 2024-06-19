from __future__ import annotations

import re
from dataclasses import dataclass
from typing import NamedTuple

from tools import loader, parsers


class Side(NamedTuple):
    mini: int
    maxi: int

    def __contains__(self, item: Side) -> bool:
        return self.mini <= item.mini and self.maxi >= item.maxi

    def intersects(self, other: Side) -> bool:
        return self.mini <= other.maxi and self.maxi >= other.mini

    def split_side(self, other: Side) -> tuple[Side, Side]:
        if self.mini < other.mini <= self.maxi:
            return Side(self.mini, other.mini - 1), Side(other.mini, self.maxi)
        if self.mini <= other.maxi < self.maxi:
            return Side(other.maxi + 1, self.maxi), Side(self.mini, other.maxi)
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
        return (self.x.intersects(other.x)
                and self.y.intersects(other.y)
                and self.z.intersects(other.z))

    def size(self) -> int:
        return ((self.x.maxi - self.x.mini + 1)
                * (self.y.maxi - self.y.mini + 1)
                * (self.z.maxi - self.z.mini + 1))

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
        op = line.split(' ')[0] == 'on'
        vals = [int(i) for i in re.findall(r'-?\d+', line)]
        if limit and (min(vals) < -50 or max(vals) > 50):
            continue
        commands.append(Cube(op=op,
                             x=Side(vals[0], vals[1]),
                             y=Side(vals[2], vals[3]),
                             z=Side(vals[4], vals[5])))
    return commands


def start(data: list[str], limit: bool) -> int:
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
                    result.extend(cube.split(command))  # noqa: B909
            else:
                break
        if command.op:
            result.append(command)
    return sum(i.size() for i in result)


print(start(parsers.lines(loader.get()), limit=True))  # 611176
print(start(parsers.lines(loader.get()), limit=False))  # 1201259791805392
