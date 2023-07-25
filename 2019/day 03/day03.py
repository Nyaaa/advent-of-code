from __future__ import annotations
from typing import NamedTuple

from tools import parsers, loader

TEST0 = """R8,U5,L5,D3
U7,R6,D4,L4"""
TEST1 = """R75,D30,R83,U83,L12,D49,R71,U7,L72
U62,R66,U55,R34,D71,R55,D58,R83"""
TEST2 = """R98,U47,R26,D63,R33,U87,L62,D20,R33,U53,R51
U98,R91,D20,R16,D67,R40,U7,R15,U6,R7"""


class Point(NamedTuple):
    row: int
    col: int

    def __add__(self, other: Point) -> Point:
        return Point(self.row + other.row, self.col + other.col)

    def manhattan_distance(self, other: Point) -> int:
        return abs(self.row - other.row) + abs(self.col - other.col)


class PCB:
    directions = {'R': Point(0, 1), 'L': Point(0, -1),
                  'U': Point(1, 0), 'D': Point(-1, 0)}

    def __init__(self, data: list):
        self.wires = []
        for line in data:
            new_wire = list()
            current_point = Point(0, 0)
            for turn in line.split(','):
                for _ in range(int(turn[1:])):
                    current_point += self.directions[turn[:1]]
                    new_wire.append(current_point)
            self.wires.append(new_wire)

    def start(self, part2: bool) -> int:
        """
        >>> print(PCB(parsers.inline_test(TEST1)).start(False))
        159

        >>> print(PCB(parsers.inline_test(TEST2)).start(False))
        135

        >>> print(PCB(parsers.inline_test(TEST1)).start(True))
        610

        >>> print(PCB(parsers.inline_test(TEST2)).start(True))
        410"""
        start = Point(0, 0)
        lowest_distance = float('inf')
        for i in set(self.wires[0]).intersection(self.wires[1]):
            if not part2:
                curr_distance = start.manhattan_distance(i)
            else:
                curr_distance = self.wires[0].index(i) + self.wires[1].index(i) + 2
            if curr_distance < lowest_distance and curr_distance != 0:
                lowest_distance = curr_distance
        return lowest_distance


print(PCB(parsers.lines(loader.get())).start(part2=False))  # 273
print(PCB(parsers.lines(loader.get())).start(part2=True))  # 15622