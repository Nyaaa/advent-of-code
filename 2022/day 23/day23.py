from tools import parsers, loader, timer
import numpy as np
from itertools import cycle
from collections import Counter
from typing import NamedTuple


class Loc(NamedTuple):
    row: int
    col: int


class Grove:
    def __init__(self, data):
        self._options = cycle(('n', 's', 'w', 'e'))
        self.map = np.array([list(0 if i == '.' else 1 for i in row) for row in data])

    def plan(self, elf: Loc, direction: str) -> Loc | None:
        row = elf.row
        col = elf.col
        sides = {'n': np.any(self.map[row - 1:row, col - 1:col + 2]),
                 's': np.any(self.map[row + 1:row + 2, col - 1:col + 2]),
                 'w': np.any(self.map[row - 1:row + 2, col - 1:col]),
                 'e': np.any(self.map[row - 1:row + 2, col + 1:col + 2])
                 }
        if all(sides.values()) or not any(sides.values()):
            return None
        else:
            return self.decide(elf, direction, sides)

    def decide(self, elf: Loc, direction: str, sides: dict[str: bool]) -> Loc:
        row = elf.row
        col = elf.col
        side = sides[direction]

        match direction:
            case 'n':
                intention = Loc(row=(row - 1), col=col) if not side else self.decide(elf, 's', sides)
            case 's':
                intention = Loc(row=(row + 1), col=col) if not side else self.decide(elf, 'w', sides)
            case 'w':
                intention = Loc(row=row, col=(col - 1)) if not side else self.decide(elf, 'e', sides)
            case _:
                intention = Loc(row=row, col=(col + 1)) if not side else self.decide(elf, 'n', sides)

        return intention

    def move(self, elf, intention):
        self.map[elf.row][elf.col] = 0
        self.map[intention.row][intention.col] = 1

    def trim(self):
        ones = np.where(self.map == 1)
        trimmed = self.map[min(ones[0]): max(ones[0]) + 1, min(ones[1]): max(ones[1]) + 1]
        return trimmed

    def get_elves(self):
        self.map = np.pad(self.map, pad_width=1, mode='constant', constant_values=0)
        positions = np.where(self.map == 1)
        return [Loc(i, j) for i, j in zip(*positions)]

    def part_1(self):
        """test part 1:
        >>> print(Grove(parsers.lines('test1.txt')).part_1())
        110
        >>> print(Grove(parsers.lines('test2.txt')).part_1())
        812"""
        counter = 1
        while counter <= 10:
            new_choice = next(self._options)
            counter += 1
            moving = {}
            for elf in self.get_elves():
                intention = self.plan(elf, new_choice)
                if intention:
                    moving[elf] = intention
            for elf in moving:
                if Counter(moving.values())[moving[elf]] == 1:
                    self.move(elf, moving[elf])
        trimmed = self.trim()
        return np.count_nonzero(trimmed == 0)  # count zeros

    def part_2(self):
        """test part 2:
        >>> print(Grove(parsers.lines('test1.txt')).part_2())
        20
        >>> print(Grove(parsers.lines('test2.txt')).part_2())
        183"""
        counter = 0
        while True:
            new_choice = next(self._options)
            counter += 1
            moving = {}
            for elf in self.get_elves():
                intention = self.plan(elf, new_choice)
                if intention:
                    moving[elf] = intention
            if not moving:
                return counter
            for elf in moving:
                if Counter(moving.values())[moving[elf]] == 1:
                    self.move(elf, moving[elf])


with timer.context():
    print(Grove(parsers.lines(loader.get())).part_1())  # 3757
with timer.context():
    print(Grove(parsers.lines(loader.get())).part_2())  # 918
