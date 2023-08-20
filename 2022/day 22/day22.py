from tools import parsers, loader
import re
from typing import NamedTuple
import numpy as np

np.set_printoptions(threshold=np.inf, linewidth=np.inf)
DIRECTIONS = {'up': ('left', 'right', 3),
              'right': ('up', 'down', 0),
              'down': ('right', 'left', 1),
              'left': ('down', 'up', 2)}


class Point(NamedTuple):
    col: int
    row: int


class Cube:
    def __init__(self, data, part2=False):
        self.part2 = part2
        _map = [list(i) for i in data[0]]
        path = re.split(r"(\d+)([A-Z])", *data[1])
        self.path = iter(list(filter(None, path)))
        self.direction = 'right'
        max_len = np.max([len(a) for a in _map])
        self.map = np.asarray([np.pad(a, (0, max_len - len(a)), 'constant', constant_values=' ') for a in _map])
        self.location = Point(np.nonzero(self.map[0] == '.')[0][0], 0)

    def warp(self, row: int, col: int) -> Point:
        match self.direction:
            case 'right': col = np.nonzero(self.map[row])[0][0]
            case 'left': col = np.nonzero(self.map[row])[0][-1]
            case 'up': row = np.nonzero(self.map[:, col])[0][-1]
            case 'down': row = np.nonzero(self.map[:, col])[0][0]
        return Point(col, row)

    def shift(self, row: int, col: int) -> tuple[Point, str]:
        """Hardcoded rotations because screw this puzzle.

            |1|2|
            |3|
          |5|4|
          |6|

        """
        if 0 <= row <= 49 and 50 <= col <= 99: face = 1
        elif 0 <= row <= 49 and 100 <= col <= 149: face = 2
        elif 50 <= row <= 99 and 50 <= col <= 99: face = 3
        elif 100 <= row <= 149 and 50 <= col <= 99: face = 4
        elif 100 <= row <= 149 and 0 <= col <= 49: face = 5
        elif 150 <= row <= 199 and 0 <= col <= 49: face = 6
        else: raise IndexError('Point is not on any of the faces.')

        # positions relative to face
        rel_row, rel_col = row % 50, col % 50

        match (face, self.direction):
            case(1, 'left'): mv = Point(0, 149 - rel_row), 'right'  # -> face 5
            case(1, 'up'): mv = Point(0, rel_col + 150), 'right'  # -> face 6

            case (2, 'up'): mv = Point(rel_col, 199), 'up'  # -> face 6
            case (2, 'right'): mv = Point(99, 149 - rel_row), 'left'  # -> face 4
            case (2, 'down'): mv = Point(99, rel_col + 50), 'left'  # -> face 3

            case (3, 'left'): mv = Point(rel_row, 100), 'down'  # -> face 5
            case (3, 'right'): mv = Point(rel_row + 100, 49), 'up'  # -> face 2

            case (4, 'down'): mv = Point(49, rel_col + 150), 'left'  # -> face 6
            case (4, 'right'): mv = Point(149, 49 - rel_row), 'left'  # -> face 2

            case (5, 'left'): mv = Point(50, 49 - rel_row), 'right'  # -> face 1
            case (5, 'up'): mv = Point(50, rel_col + 50), 'right'  # -> face 3

            case (6, 'right'): mv = Point(rel_row + 50, 149), 'up'  # -> face 4
            case (6, 'down'): mv = Point(rel_col + 100, 0), 'down'  # -> face 2
            case (6, 'left'): mv = Point(rel_row + 50, 0), 'down'  # -> face 1

            case _:
                self.draw()
                raise NotImplementedError(f'No transition {self.direction} from face {face}.')
        return mv

    def move(self, distance: int):
        # print(distance, self.direction)

        for _ in range(distance):
            col, row = self.location
            current_location = self.location
            current_direction = self.direction

            match self.direction:
                case 'right': col += 1
                case 'left': col -= 1
                case 'up': row -= 1
                case 'down': row += 1

            try:
                cell = self.map[row][col]
            except IndexError:
                cell = ' '

            match (cell, self.part2):
                case (' ', False): _next = self.warp(row, col)
                case (' ', True): _next, self.direction = self.shift(current_location.row, current_location.col)
                case _: _next = Point(col, row)

            match self.map[_next.row][_next.col]:
                case '.' | 'X': self.location = _next
                case '#': self.location, self.direction = current_location, current_direction

            self.map[self.location.row][self.location.col] = 'X'

    def start(self):
        """test part 1:
        >>> c = Cube(parsers.blocks('test.txt'))
        >>> print(c.start())
        6032
        >>> c = Cube(parsers.blocks('test.txt'))
        >>> c.path = iter([1, 'L', 1, 'L', 1, 'L', 1, 'L'])
        >>> print(c.start())
        1036
        """
        while True:
            try:
                move = int(next(self.path))
                self.move(move)
                turn = next(self.path)
                turn = 1 if turn == 'R' else 0
                self.direction = DIRECTIONS[self.direction][turn]
            except StopIteration:
                col = self.location.col + 1
                row = self.location.row + 1
                direction = DIRECTIONS[self.direction][2]
                # print(self.map)
                return (1000 * row) + (4 * col) + direction


print(Cube(parsers.blocks(loader.get())).start())  # 76332
print(Cube(parsers.blocks(loader.get()), part2=True).start())  # 144012
