from tools import parsers, loader
import numpy as np
from itertools import cycle
from collections import Counter, namedtuple


class Elf:
    def __init__(self, location):
        self.location = location
        self.intention = None


class Grove:
    def __init__(self, data):
        self.loc = namedtuple('Loc', 'row, col')
        self._options = cycle(('n', 's', 'w', 'e'))
        self.map = np.array([list(0 if i == '.' else 1 for i in row) for row in data])
        self.map = np.pad(self.map, pad_width=100, mode='constant', constant_values=0)
        positions = np.where(self.map == 1)
        locations = [self.loc(i, j) for i, j in zip(*positions)]
        self.elves = [Elf(loc) for loc in locations]

    def plan(self, elf: Elf, direction: str):
        row = elf.location.row
        col = elf.location.col
        north = np.any(self.map[row - (row != 0):row, col - (col != 0):col + 2])
        south = np.any(self.map[row + 1:row + 2, col - (col != 0):col + 2])
        west = np.any(self.map[row - (row != 0):row + 2, col - (col != 0):col])
        east = np.any(self.map[row - (row != 0):row + 2, col + 1:col + 2])
        sides = (north, south, west, east)
        if all(sides) or not any(sides):
            return None

        if direction == 'n':
            elf.intention = self.loc(row=(row - 1), col=col) if not north else self.plan(elf, 's')
        elif direction == 's':
            elf.intention = self.loc(row=(row + 1), col=col) if not south else self.plan(elf, 'w')
        elif direction == 'w':
            elf.intention = self.loc(row=row, col=(col - 1)) if not west else self.plan(elf, 'e')
        elif direction == 'e':
            elf.intention = self.loc(row=row, col=(col + 1)) if not east else self.plan(elf, 'n')

        return elf.intention

    def move(self, elf: Elf):
        self.map[elf.location.row][elf.location.col] = 0
        self.map[elf.intention.row][elf.intention.col] = 1
        elf.location = elf.intention

    def trim(self):
        ones = np.where(self.map == 1)
        trimmed = self.map[min(ones[0]): max(ones[0]) + 1, min(ones[1]): max(ones[1]) + 1]
        return trimmed

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
            for elf in self.elves:
                intention = self.plan(elf, new_choice)
                if intention:
                    moving[elf] = intention
            for elf in moving:
                if Counter(moving.values())[elf.intention] == 1:
                    self.move(elf)
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
            for elf in self.elves:
                intention = self.plan(elf, new_choice)
                if intention:
                    moving[elf] = intention
            if not moving:
                return counter
            for elf in moving:
                if Counter(moving.values())[elf.intention] == 1:
                    self.move(elf)


print(Grove(parsers.lines(loader.get())).part_1())  # 3757
print(Grove(parsers.lines(loader.get())).part_2())  # 918
