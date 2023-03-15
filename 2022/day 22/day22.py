from tools import parsers, loader
import re
from collections import namedtuple


class Cube:
    DIRECTIONS = {'up': ('left', 'right'),
                  'right': ('up', 'down'),
                  'down': ('right', 'left'),
                  'left': ('down', 'up')}

    def __init__(self, data):
        self.map = [list(i) for i in data[0]]
        path = re.split(r"(\d+)([A-Z])", *data[1])
        self.path = parsers.generator(list(filter(None, path)))
        self.point = namedtuple('Location', 'Column, Row')
        self.location = self.point(self.map[0].index('.'), 0)  # column, row
        self.direction = 'right'

    def move(self, distance: int, turn: str):
        if turn == 'R':
            turn = 1
        else:
            turn = 0

        for _ in range(distance):
            row = self.location[1]
            col = self.location[0]
            current = self.point(col, row)

            match self.direction:
                case 'right':
                    next = self.point(col + 1, row)
                case 'left':
                    next = self.point(col - 1, row)
                case 'up':
                    next = self.point(col, row - 1)
                case 'down':
                    next = self.point(col, row + 1)

            cell = self.map[next[1]][next[0]]

            match cell:
                case '.':
                    self.location = next
                case '#':
                    self.location = current
                case ' ':
                    raise NotImplementedError

        self.direction = self.DIRECTIONS[self.direction][turn]
        print(self.location)

    def part_1(self):
        while True:
            move = int(next(self.path))
            turn = next(self.path)
            print(move, turn)
            self.move(move, turn)


print(Cube(parsers.blocks('test.txt')).part_1())
# print(Cube(parsers.blocks(loader.get())))

