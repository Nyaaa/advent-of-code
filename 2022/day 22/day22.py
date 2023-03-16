from tools import parsers, loader
import re
from collections import namedtuple
import numpy as np

np.set_printoptions(threshold=np.inf, linewidth=np.inf)


class Cube:
    DIRECTIONS = {'up': ('left', 'right', 3),
                  'right': ('up', 'down', 0),
                  'down': ('right', 'left', 1),
                  'left': ('down', 'up', 2)}

    def __init__(self, data):
        _map = [list(i) for i in data[0]]
        path = re.split(r"(\d+)([A-Z])", *data[1])
        self.path = parsers.generator(list(filter(None, path)))
        self.point = namedtuple('Location', 'Column, Row')
        self.direction = 'right'
        max_len = np.max([len(a) for a in _map])
        self.map = np.asarray([np.pad(a, (0, max_len - len(a)), 'constant', constant_values=' ') for a in _map])
        self.location = self.point(np.nonzero(self.map[0] == '.')[0][0], 0)

    def warp(self, row: int, col: int):
        match self.direction:
            case 'right':
                col = np.nonzero(self.map[row])[0][0]
            case 'left':
                col = np.nonzero(self.map[row])[0][-1]
            case 'up':
                row = np.nonzero(self.map[:, col])[0][-1]
            case 'down':
                row = np.nonzero(self.map[:, col])[0][0]

        return self.point(col, row)

    def move(self, distance: int):
        # print(distance, self.direction)

        for _ in range(distance):
            col, row = self.location
            current = self.point(col, row)

            match self.direction:
                case 'right':
                    col += 1
                case 'left':
                    col -= 1
                case 'up':
                    row -= 1
                case 'down':
                    row += 1

            try:
                cell = self.map[row][col]
            except IndexError:
                cell = ' '

            if cell == ' ':
                _next = self.warp(row, col)
            else:
                _next = self.point(col, row)
            cell = self.map[_next.Row][_next.Column]

            if cell == '.' or cell == 'X':
                self.location = _next
            elif cell == '#':
                self.location = current

            # self.map[row][col] = 'X'

        # print(self.location, self.direction)
        # print(self.map[row-2:row+2, col-2:col+2])

    def part_1(self):
        """test part 1:
        >>> c = Cube(parsers.blocks('test.txt'))
        >>> print(c.part_1())
        6032
        >>> c = Cube(parsers.blocks('test.txt'))
        >>> c.path = parsers.generator([1, 'L', 1, 'L', 1, 'L', 1, 'L'])
        >>> print(c.part_1())
        1036
        """
        while True:
            try:
                move = int(next(self.path))
                self.move(move)
                turn = next(self.path)
                turn = 1 if turn == 'R' else 0
                self.direction = self.DIRECTIONS[self.direction][turn]
            except StopIteration:
                # for i in range(len(self.map)):
                #     print(self.map[i])
                col = self.location.Column + 1
                row = self.location.Row + 1
                direction = self.DIRECTIONS[self.direction][2]
                # print(self.location, direction)
                return (1000 * row) + (4 * col) + direction


print(Cube(parsers.blocks(loader.get())).part_1())  # 76332

# print(Cube(parsers.blocks('test.txt')).part_1())
