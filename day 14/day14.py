from tools import parsers
import numpy as np
from tabulate import tabulate

test = """498,4 -> 498,6 -> 496,6
503,4 -> 502,4 -> 502,9 -> 494,9
"""
# data = parsers.inline_test(test)
data = parsers.lines('input14.txt')
data_clean = [[[int(i) for i in coord.split(',')] for coord in line.split(' -> ')] for line in data]
np.set_printoptions(threshold=np.inf, linewidth=200)


class Cave:
    def __init__(self, part: int):
        self.left, self.right = 500, 501
        self.cavern = np.chararray((1, 700), unicode=True)  # TODO make dynamic adjustments instead of hardcoded borders
        self.cavern[:] = ' '
        self.start = 500
        self.counter = 0

        for line in data_clean:
            for i in range(len(line)):
                x, y = line[i][0], line[i][1]
                try:
                    next_x, next_y = line[i + 1][0], line[i + 1][1]
                except IndexError:
                    break
                max_x, min_x, max_y = max(x, next_x), min(x, next_x), max(y, next_y)
                self.left = min_x if min_x < self.left else self.left
                self.right = max_x if max_x > self.right else self.right
                delta_x = x - next_x
                if max_y > len(self.cavern):
                    add = max_y + 1 - len(self.cavern)
                    self.increase_depth(add, 'zeros')

                if delta_x > 0:
                    for _x in range(next_x, x + 1):
                        self.cavern[y, _x] = '█'
                if delta_x < 0:
                    for _x in range(x, next_x + 1):
                        self.cavern[y, _x] = '█'

                if y - next_y > 0:
                    for _y in range(next_y, y + 1):
                        self.cavern[_y, x] = '█'
                elif y - next_y < 0:
                    for _y in range(y, next_y + 1):
                        self.cavern[_y, x] = '█'

        if part == 2:
            self.increase_depth(1, 'zeros')
            self.increase_depth(1, 'ones')

    def increase_depth(self, lines, char):
        if char == 'zeros':
            newline = np.chararray((lines, 700), unicode=True)
            newline[:] = ' '
        else:
            newline = np.chararray((lines, 700), unicode=True)
            newline[:] = '█'
        self.cavern = np.append(self.cavern, newline, axis=0)

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

    def go(self):
        while True:
            try:
                self.fall(0, self.start)
                self.counter += 1
            except IndexError:
                break


# part 1
c = Cave(part=1)
c.go()
cropped = c.cavern[:, c.left - 1:c.right + 1]
print(tabulate(cropped))
print(c.counter)  # 774

# part 2
c2 = Cave(part=2)
c2.go()
cropped2 = c2.cavern[:, c2.left - 1:c2.right + 1]
print(tabulate(cropped2))
print(c2.counter)  # 22499
