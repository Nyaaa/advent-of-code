from tools import parsers, loader
import numpy as np
from tabulate import tabulate

test = """498,4 -> 498,6 -> 496,6
503,4 -> 502,4 -> 502,9 -> 494,9
"""
np.set_printoptions(threshold=np.inf, linewidth=200)


class Cave:
    def __init__(self, data, part: int):
        """Right border is hardcoded as an arbitrarily large value,
        since adding dynamic stretching adds extra complexity to the code. May need to increase it if code fails."""
        self.left, self.right = 500, 700
        self.cavern = np.chararray((1, self.right), unicode=True)
        self.cavern[:] = ' '
        _data_clean = [[[int(i) for i in coord.split(',')] for coord in line.split(' -> ')] for line in data]

        for line in _data_clean:
            for i, (x, y) in enumerate(line):
                try:
                    next_x, next_y = line[i + 1][0], line[i + 1][1]
                except IndexError:
                    break
                max_x, min_x, max_y = max(x, next_x), min(x, next_x), max(y, next_y)
                self.left = min_x if min_x < self.left else self.left
                self.right = max_x if max_x > self.right else self.right
                if max_y > len(self.cavern):
                    add = max_y + 1 - len(self.cavern)
                    self.increase_depth(add, ' ')

                if x - next_x > 0:
                    for _x in range(next_x, x + 1):
                        self.cavern[y, _x] = '█'
                elif x - next_x < 0:
                    for _x in range(x, next_x + 1):
                        self.cavern[y, _x] = '█'

                if y - next_y > 0:
                    for _y in range(next_y, y + 1):
                        self.cavern[_y, x] = '█'
                elif y - next_y < 0:
                    for _y in range(y, next_y + 1):
                        self.cavern[_y, x] = '█'

        if part == 2:
            self.increase_depth(1, ' ')
            self.increase_depth(1, '█')

    def increase_depth(self, lines, char):
        self.cavern = np.pad(self.cavern, [(0, lines), (0, 0)], mode='constant', constant_values=char)

    def fall(self, row: int, column: int):
        if self.cavern[0][column] == 'S':
            raise IndexError('reached the top')
        if self.cavern[row][column] not in ['█', 'S']:
            return self.fall(row + 1, column)
        else:
            if self.cavern[row][column - 1] not in ['█', 'S']:
                return self.fall(row, column - 1)
            else:
                if self.cavern[row][column + 1] not in ['█', 'S']:
                    return self.fall(row, column + 1)
                else:
                    self.cavern[row - 1][column] = 'S'
                    self.left = column if column < self.left else self.left
                    self.right = column if column > self.right else self.right

    def go(self, vis=False):
        """test part 1:
        >>> print(Cave(parsers.inline_test(test), part=1).go())
        24

        test part 2:
        >>> print(Cave(parsers.inline_test(test), part=2).go())
        93"""
        counter = 0
        while True:
            try:
                self.fall(0, 500)
                counter += 1
            except IndexError:
                break
        if vis:
            cropped = self.cavern[:, self.left - 1:self.right + 1]
            print(tabulate(cropped))
        return counter


# part 1
print(Cave(parsers.lines(loader.get()), part=1).go())  # 774

# part 2
print(Cave(parsers.lines(loader.get()), part=2).go())  # 22499
