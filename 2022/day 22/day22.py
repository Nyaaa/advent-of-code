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
                left = np.nonzero(self.map[row])[0][0]
                return self.point(left, row)
            case 'left':
                right = np.nonzero(self.map[row])[0][-1]
                return self.point(right, row)
            case 'up':
                down = np.nonzero(self.map[:, col])[0][-1]
                return self.point(col, down)
            case 'down':
                up = np.nonzero(self.map[:, col])[0][0]
                return self.point(col, up)

    def move(self, distance: int):
        print(distance, self.direction)

        for _ in range(distance):
            col, row = self.location
            current = self.point(col, row)

            match self.direction:
                case 'right':
                    _next = self.point(col + 1, row)
                case 'left':
                    _next = self.point(col - 1, row)
                case 'up':
                    _next = self.point(col, row - 1)
                case 'down':
                    _next = self.point(col, row + 1)

            try:
                cell = self.map[_next[1]][_next[0]]
            except IndexError:
                cell = ' '

            if cell == ' ':
                _next = self.warp(row, col)
                cell = self.map[_next[1]][_next[0]]

            if cell == '.':
                self.location = _next
            elif cell == '#':
                self.location = current

            self.map[row][col] = 'X'

        print(self.location, self.direction)

    def part_1(self):
        while True:
            try:
                move = int(next(self.path))
                self.move(move)
                turn = next(self.path)
                turn = 1 if turn == 'R' else 0
                self.direction = self.DIRECTIONS[self.direction][turn]
            except StopIteration:
                for i in range(len(self.map)):
                    print(self.map[i])
                col = self.location[0] + 1
                row = self.location[1] + 1
                direction = self.DIRECTIONS[self.direction][2]
                print(self.location)
                print(row, col, direction)
                return (1000 * row) + (4 * col) + direction


print(Cube(parsers.blocks('test.txt')).part_1())
print(Cube(parsers.blocks(loader.get())).part_1())

# 111144 too high
# Location(Column=35, Row=110)
