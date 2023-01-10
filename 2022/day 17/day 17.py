from itertools import cycle
from tools import parsers
import numpy as np

test = '>>><<><>><<<>><>>><<<>>><<<><<<>><>><<>>'

LEFT = 0
RIGHT = 7
STONES = [[(0, 2), (0, 3), (0, 4), (0, 5)],
          [(1, 2), (1, 3), (1, 4), (0, 3), (2, 3)],
          [(2, 2), (2, 3), (2, 4), (1, 4), (0, 4)],
          [(0, 2), (1, 2), (2, 2), (3, 2)],
          [(0, 2), (0, 3), (1, 2), (1, 3)]]


class Cave:
    def __init__(self, jets):
        self.cavern = np.chararray((1, RIGHT), unicode=True)
        self.cavern[:] = '█'
        self.stones = cycle(STONES)
        self.jets = cycle(jets)

    def launch(self, stone):  # too high?
        for i in range(0, 4):
            place = self.can_fall(stone, i)
            if not place:
                newline = np.chararray((1, RIGHT), unicode=True)
                newline[:] = '_'
                self.cavern = np.insert(self.cavern, 0, newline, axis=0)
                print(self.cavern)
                self.launch(stone)

    def move(self, stone, direction):
        print('falling', direction)
        _stone = []
        for unit in stone:
            if direction == 'down':
                _unit = (unit[0] + 1, unit[1])
                _stone.append(_unit)
                if self.cavern[unit[0] + 1][unit[1]] == '█':
                    return stone
            if direction == '>':
                _unit = (unit[0], unit[1] + 1)
                _stone.append(_unit)
                if unit[1] + 1 >= RIGHT:
                    return stone
                if self.cavern[unit[0]][unit[1] + 1] == '█':
                    return stone
            elif direction == '<':
                _unit = (unit[0], unit[1] - 1)
                _stone.append(_unit)
                if unit[1] - 1 < LEFT:
                    return stone
                if self.cavern[unit[0]][unit[1] - 1] == '█':
                    return stone
        return _stone

    def fall(self, stone):
        print(stone)
        jet = next(self.jets)
        stone = self.move(stone, jet)
        print(stone)
        new_stone = self.move(stone, 'down')
        if stone != new_stone:
            return self.fall(new_stone)
        else:
            for unit in stone:
                row, column = unit
                self.cavern[row][column] = '█'

    def can_fall(self, stone, rows: int = 1) -> bool:
        result = True
        for i in range(0, rows+1):
            for unit in stone:
                row, col = unit
                if '█' in self.cavern[row + i]:
                    result = False
                    break
        return result

    def part_1(self):
        rocks = 5

        while rocks > 0:
            stone = next(self.stones)
            self.launch(stone)
            self.fall(stone)
            print(self.cavern)
            rocks -= 1


print(Cave(test).part_1())  # fails
