from typing import Optional
from tools import parsers, loader
import numpy as np
from dataclasses import dataclass


@dataclass
class Loc:
    row: int
    col: int


@dataclass
class Elf:
    location: Loc
    intention: Optional[Loc] = None


class Grove:
    def __init__(self, data):
        self.map = np.array([list(0 if i == '.' else 1 for i in row) for row in data])
        print(self.map)
        positions = np.where(self.map == 1)
        locations = [Loc(i, j) for i, j in zip(*positions)]
        self.elves = [Elf(loc) for loc in locations]

    def plan(self, elf: Elf) -> Loc:
        row = elf.location.row
        col = elf.location.col
        print('location', elf.location)
        north = self.map[row - (row != 0):row, col - (col != 0):col + 2].flatten()
        south = self.map[row + 1:row + 2, col - (col != 0):col + 2].flatten()
        west = self.map[row - (row != 0):row + 2, col - (col != 0):col].flatten()
        east = self.map[row - (row != 0):row + 2, col + 1:col + 2].flatten()

        if 1 not in (*north, *south, *west, *east):
            elf.intention = elf.location
            print('nop')

        if 1 not in north:
            elf.intention = Loc(row=(row - 1), col=col)
            print('n')
        elif 1 not in south:
            elf.intention = Loc(row=(row + 1), col=col)
            print('s')
        elif 1 not in west:
            elf.intention = Loc(row=row, col=(col - 1))
            print('w')
        elif 1 not in east:
            elf.intention = Loc(row=row, col=(col + 1))
            print('e')
        print('intent', elf.intention)
        return elf.intention

    def part_1(self):
        while True:
            # self.map = np.pad(self.map, pad_width=1, mode='constant', constant_values=0)
            intentions = []
            for elf in self.elves:
                intention = self.plan(elf)
                intentions.append(intention)
            for elf in self.elves:
                intention = self.plan(elf)
            print(intentions)
            break


print(Grove(parsers.lines('test.txt')).part_1())
