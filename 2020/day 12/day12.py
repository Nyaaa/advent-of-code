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

    def __add__(self, other):
        return Point(row=self.row + other.row, col=self.col + other.col)


class Navigation:
    def __init__(self, data: list):
        self.direction = 'E'
        self.position = Point(0, 0)
        self.instructions = []
        for i in data:
            d = i[0]
            val = int(i[1:])
            self.instructions.append((d, val))
        self.headings = deque(['N', 'W', 'S', 'E'])

    def get_move(self, heading, distance):
        if heading == 'F':
            heading = self.direction
        elif heading == 'L':
            turns = distance // 90
            while self.headings[0] != self.direction:
                self.headings.rotate()
            self.headings.rotate(-turns)
            self.direction = self.headings[0]
            return Point(0, 0)
        elif heading == 'R':
            turns = distance // 90
            while self.headings[0] != self.direction:
                self.headings.rotate()
            self.headings.rotate(turns)
            self.direction = self.headings[0]
            return Point(0, 0)

        if heading == 'N':
            move = Point(distance, 0)
        elif heading == 'S':
            move = Point(-distance, 0)
        if heading == 'E':
            move = Point(0, distance)
        elif heading == 'W':
            move = Point(0, -distance)
        return move

    def part_1(self):
        for heading, val in self.instructions:
            move = self.get_move(heading, val)
            self.position += move

        return abs(self.position[0]) + abs(self.position[1])


print(Navigation(parsers.inline_test(TEST)).part_1())
print(Navigation(parsers.lines(loader.get())).part_1())  # 858
