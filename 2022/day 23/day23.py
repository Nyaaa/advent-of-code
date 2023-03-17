from typing import Optional
from tools import parsers, loader
import numpy as np
from dataclasses import dataclass
from itertools import cycle
from collections import Counter


@dataclass
class Loc:
    row: int
    col: int

    def __hash__(self):
        return hash((self.row, self.col))


class Elf:
    def __init__(self, location):
        self._options = cycle(('n', 's', 'w', 'e'))
        self.location = location
        self.intention: Optional[Loc] = None
        self.prev_choice: Optional[str] = None

    def choice(self):
        new_choice = next(self._options)
        self.prev_choice = new_choice
        return new_choice


class Grove:
    def __init__(self, data):
        self.map = np.array([list(0 if i == '.' else 1 for i in row) for row in data])
        positions = np.where(self.map == 1)
        locations = [Loc(i, j) for i, j in zip(*positions)]
        self.elves = [Elf(loc) for loc in locations]

    def plan(self, elf: Elf, direction: str):
        print('*' * 20)
        row = elf.location.row
        col = elf.location.col
        print('location', elf.location)
        north = self.map[row - (row != 0):row, col - (col != 0):col + 2].flatten()
        south = self.map[row + 1:row + 2, col - (col != 0):col + 2].flatten()
        west = self.map[row - (row != 0):row + 2, col - (col != 0):col].flatten()
        east = self.map[row - (row != 0):row + 2, col + 1:col + 2].flatten()

        go_north = Loc(row=(row - 1), col=col)
        go_south = Loc(row=(row + 1), col=col)
        go_west = Loc(row=row, col=(col - 1))
        go_east = Loc(row=row, col=(col + 1))

        if direction == 'n':
            elf.intention = go_north if 1 not in north else self.plan(elf, 's')
        elif direction == 's':
            elf.intention = go_south if 1 not in south else self.plan(elf, 'w')
        elif direction == 'w':
            elf.intention = go_west if 1 not in west else self.plan(elf, 'e')
        elif direction == 'e':
            elf.intention = go_east if 1 not in east else self.plan(elf, 'n')

        if 1 not in (*north, *south, *west, *east):
            elf.intention = None

        print('intent', elf.intention)
        return elf.intention

    def move(self, elf: Elf):
        if elf.intention:
            self.map[elf.location.row][elf.location.col] = 0
            self.map[elf.intention.row][elf.intention.col] = 1
            elf.location = elf.intention

    def part_1(self):
        elf_count = len(self.elves)
        while True:
            # self.map = np.pad(self.map, pad_width=1, mode='constant', constant_values=0)
            moving = {}
            print('=' * 20)
            for elf in self.elves:
                direction = elf.choice()
                intention = self.plan(elf, direction)
                moving[elf] = intention
            if Counter(moving.values())[None] == elf_count:
                break
            print(Counter(moving.values()))
            for elf in moving:
                if Counter(moving.values())[elf.intention] == 1:
                    self.move(elf)
            print(self.map)


print(Grove(parsers.lines('test.txt')).part_1())
