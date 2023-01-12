from itertools import cycle
from tools import parsers
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
        self.slice = []

    def launch(self, stone, level=0):
        place = self.can_fall(stone, level)
        if level < 4:
            if not place:
                newline = np.chararray((1, RIGHT), unicode=True)
                newline[:] = '_'
                self.cavern = np.insert(self.cavern, 0, newline, axis=0)
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
                if unit[1] + 1 >= RIGHT:
                    return stone
                if self.cavern[unit[0]][unit[1] + 1] == '█':
                    return stone
            elif direction == '<':
                _stone.append((unit[0], unit[1] - 1))
                if unit[1] - 1 < LEFT:
                    return stone
                if self.cavern[unit[0]][unit[1] - 1] == '█':
                    return stone
        return _stone

    def fall(self, stone):
        jet = next(self.jets)
        stone = self.move(stone, jet)
        new_stone = self.move(stone, 'down')
        if stone != new_stone:
            return self.fall(new_stone)
        else:
            for unit in stone:
                row, column = unit
                self.cavern[row][column] = '█'

    def can_fall(self, stone, rows: int = 1) -> bool:
        for i in range(0, rows+1):
            for unit in stone:
                row, col = unit
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
        rocks = 2022

        while rocks > 0:
            stone = next(self.stones)
            self.launch(stone)
            self.fall(stone)
            rocks -= 1
        self.trim()
        return len(self.cavern)-1

    def part_2(self):
        rocks = 1000000000000
        # rocks = 2022
        matches = {}
        counter = 0
        heights = {}
        seq_length = None
        seq_stones = None
        skipped_height = None

        while rocks > counter:
            if seq_length and not skipped_height:
                _counter = counter - 1
                remains = rocks - _counter
                seqs = remains // seq_stones
                skipped_stones = seqs * seq_stones
                skipped_height = seqs * seq_length
                counter += skipped_stones
                print(seq_stones, seq_length)
                print(f'Skipped {skipped_stones} stones, {skipped_height} height')

            counter += 1
            # print(counter)
            stone = next(self.stones)
            self.launch(stone)
            self.fall(stone)

            window = 30
            top = self.cavern[10:40]
            old = 40

            # rolling window, 2D array
            if len(matches) < 2 and not seq_length:
                for i in range(0, self.cavern.shape[0] - window + 1):
                    _window = self.cavern[i:i + window, :]
                    result = np.array_equal(top, _window)
                    bottom = i + window

                    if result and old != bottom:
                        # print(old, bottom, counter)
                        if not matches.get(old):
                            matches[old] = counter
                        if not heights.get(old):
                            self.trim()
                            heights[old] = len(self.cavern) - 1
                        old = bottom

            else:
                seq_length = int(np.diff(list(matches.keys())))
                seq_stones = int(np.diff(list(matches.values())))


        print(matches)
        print(heights)

        self.trim()
        return len(self.cavern) - 1 + skipped_height


def rolling_window(a, shape):
    s = (a.shape[0] - shape[0] + 1,) + (a.shape[1] - shape[1] + 1,) + shape
    strides = a.strides + a.strides
    return np.lib.stride_tricks.as_strided(a, shape=s, strides=strides)


# print(Cave(test).part_1())  # 3068
# print(Cave(*parsers.lines('input.txt')).part_1())  # 3059
# print(Cave(test).part_2())  # 1514285714288
print(Cave(*parsers.lines('input.txt')).part_2())  # 1514450867056 too high

"""DEBUG:
1730 2620
Skipped 999999994670 stones, 1514450858980 height
{40: 2022, 2660: 3752}
{40: 3059, 2660: 5679}
1514450867056
"""