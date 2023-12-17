from collections import Counter
from itertools import cycle

import numpy as np

from tools import common, loader, parsers, timer


class Grove:
    def __init__(self, data: list[str]) -> None:
        self._options = cycle(('n', 's', 'w', 'e'))
        self.map = np.array([[0 if i == '.' else 1 for i in row] for row in data])

    def plan(self, elf: tuple, direction: str) -> tuple | None:
        row, col = elf
        sides = {'n': np.any(self.map[row - 1:row, col - 1:col + 2]),
                 's': np.any(self.map[row + 1:row + 2, col - 1:col + 2]),
                 'w': np.any(self.map[row - 1:row + 2, col - 1:col]),
                 'e': np.any(self.map[row - 1:row + 2, col + 1:col + 2])
                 }
        if all(sides.values()) or not any(sides.values()):
            return None
        return self.decide(elf, direction, sides)

    def decide(self, elf: tuple, direction: str, sides: dict[str: bool]) -> tuple:
        side = sides[direction]
        match direction:
            case 'n':
                intention = (elf[0] - 1, elf[1]) if not side else self.decide(elf, 's', sides)
            case 's':
                intention = (elf[0] + 1, elf[1]) if not side else self.decide(elf, 'w', sides)
            case 'w':
                intention = (elf[0], elf[1] - 1) if not side else self.decide(elf, 'e', sides)
            case _:
                intention = (elf[0], elf[1] + 1) if not side else self.decide(elf, 'n', sides)

        return intention

    def move(self, elf: tuple, intention: tuple) -> None:
        self.map[elf] = 0
        self.map[intention] = 1

    def get_elves(self) -> list[tuple]:
        self.map = np.pad(self.map, pad_width=1, mode='constant', constant_values=0)
        positions = np.where(self.map == 1)
        return list(zip(*positions, strict=True))

    def part_1(self) -> int:
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
        trimmed = common.trim_array(self.map)
        return np.count_nonzero(trimmed == 0)

    def part_2(self) -> int:
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
