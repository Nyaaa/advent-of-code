from itertools import cycle
from tools import parsers, loader, timer
import numpy as np
np.set_printoptions(threshold=np.inf, linewidth=100)

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

    def launch(self, stone, level=0):
        place = self.can_fall(stone, level)
        if level < 4:
            if not place:
                # numpy padding: [(top, bottom), (left, right)] value is thickness
                self.cavern = np.pad(self.cavern, [(1, 0), (0, 0)], mode='constant', constant_values='_')
        elif level >= 4 and place:
            self.cavern = np.delete(self.cavern, 0, 0)
            self.launch(stone, level)
        else:
            return
        self.launch(stone, level + 1)

    def move(self, stone, direction):
        _stone = []
        for unit in stone:
            if direction == 'down':
                _stone.append((unit[0] + 1, unit[1]))
                if self.cavern[unit[0] + 1][unit[1]] == '█':
                    return stone
            if direction == '>':
                _stone.append((unit[0], unit[1] + 1))
                if unit[1] + 1 >= RIGHT or self.cavern[unit[0]][unit[1] + 1] == '█':
                    return stone
            elif direction == '<':
                _stone.append((unit[0], unit[1] - 1))
                if unit[1] - 1 < LEFT or self.cavern[unit[0]][unit[1] - 1] == '█':
                    return stone
        return _stone

    def fall(self, stone):
        jet = next(self.jets)
        stone = self.move(stone, jet)
        new_stone = self.move(stone, 'down')
        if stone != new_stone:
            return self.fall(new_stone)
        else:
            for row, column in stone:
                self.cavern[row][column] = '█'

    def can_fall(self, stone, rows: int = 1) -> bool:
        for i in range(0, rows+1):
            for row, col in stone:
                try:
                    if '█' in self.cavern[row + i]:
                        return False
                except IndexError:
                    return False
        return True

    def trim(self):
        while True:
            if '█' not in self.cavern[0]:
                self.cavern = np.delete(self.cavern, 0, 0)
            else:
                break

    def part_1(self):
        """test part 1:
        >>> print(Cave(test).part_1())
        3068"""
        rocks = 2022

        while rocks > 0:
            stone = next(self.stones)
            self.launch(stone)
            self.fall(stone)
            rocks -= 1
        self.trim()
        return len(self.cavern) - 1

    def part_2(self, rocks):
        """test part 1:
        >>> print(Cave(test).part_2(2022))
        3068

        test part 2:
        >>> print(Cave(test).part_2(1000000000000))
        1514285714288"""
        matches = {}
        counter = 0
        seq_length = None  # test: 53, input: 2574
        seq_stones = None  # test: 35, input: 1715
        skipped_height = None

        while rocks > counter:
            counter += 1
            stone = next(self.stones)
            self.launch(stone)
            self.fall(stone)

            # increase the window size if fails
            top = self.cavern[5:25]

            if not seq_length:
                found = np.all(np.all(rolling_window(self.cavern, top.shape) == top, axis=2), axis=2).nonzero()
                rez = found[0]
                if 3 >= len(rez) > 1 and rez[-1] not in matches:
                    matches[rez[-1]] = counter
                if len(matches) == 2:
                    seq_length = int(np.diff(list(matches.keys())))
                    seq_stones = int(np.diff(list(matches.values())))
                    remains = rocks - counter - 1
                    seqs = remains // seq_stones
                    skipped_stones = seqs * seq_stones
                    skipped_height = seqs * seq_length
                    counter += skipped_stones

        self.trim()
        return len(self.cavern) - 1 + skipped_height


def rolling_window(a, shape):
    """Rolling window for 2D numpy array"""
    s = (a.shape[0] - shape[0] + 1,) + (a.shape[1] - shape[1] + 1,) + shape
    strides = a.strides + a.strides
    return np.lib.stride_tricks.as_strided(a, shape=s, strides=strides)


with timer.context():
    print(Cave(*parsers.lines(loader.get())).part_1())  # 3059
with timer.context():
    print(Cave(*parsers.lines(loader.get())).part_2(1000000000000))  # 1500874635587
