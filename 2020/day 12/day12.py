from __future__ import annotations
from collections import deque
from typing import NamedTuple
from tools import parsers, loader

TEST = """F10
N3
F7
R90
F11
"""


class Point(NamedTuple):
    row: int
    col: int

    def __add__(self, other: Point):
        return Point(row=self.row + other.row, col=self.col + other.col)

    def __mul__(self, other: int):
        return Point(row=self.row * other, col=self.col * other)

    def __repr__(self):
        return f'({"East" if self.col > 0 else "West"} {abs(self.col)}, ' \
               f'{"North" if self.row > 0 else "South"} {abs(self.row)})'


class Navigation:
    def __init__(self, data: list):
        self.direction = 'E'
        self.position = Point(0, 0)
        self.waypoint = Point(1, 10)
        self.instructions = [(i[0], int(i[1:])) for i in data]
        self.headings = deque(['N', 'W', 'S', 'E'])

    def turn(self, direction, value):
        turns = value // 90
        while self.headings[0] != self.direction:
            self.headings.rotate()
        self.headings.rotate(-turns if direction == 'L' else turns)
        self.direction = self.headings[0]
        return Point(0, 0)

    def get_move(self, heading, distance):
        match heading:
            case 'F': move = self.get_move(self.direction, distance)
            case 'N': move = Point(distance, 0)
            case 'S': move = Point(-distance, 0)
            case 'E': move = Point(0, distance)
            case 'W': move = Point(0, -distance)
            case _: move = self.turn(heading, distance)
        return move

    def part_1(self):
        """test part 1:
        >>> print(Navigation(parsers.inline_test(TEST)).part_1())
        25"""
        for heading, val in self.instructions:
            self.position += self.get_move(heading, val)
        return abs(self.position[0]) + abs(self.position[1])

    def rotate(self, value):
        x, y = self.waypoint
        rot = {90: Point(-y, x),
               180: Point(-x, -y),
               270: Point(y, -x)}
        return rot[value]

    def part_2(self):
        """test part 2:
        >>> print(Navigation(parsers.inline_test(TEST)).part_2())
        286"""
        for heading, val in self.instructions:
            match heading:
                case 'F': self.position += self.waypoint * val
                case 'R': self.waypoint = self.rotate(val)
                case 'L': self.waypoint = self.rotate(360 - val)
                case _: self.waypoint += self.get_move(heading, val)
        return abs(self.position[0]) + abs(self.position[1])


print(Navigation(parsers.lines(loader.get())).part_1())  # 858
print(Navigation(parsers.lines(loader.get())).part_2())  # 39140
