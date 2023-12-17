import re

import numpy as np

from tools import loader, parsers

np.set_printoptions(threshold=np.inf, linewidth=np.inf)
DIRECTIONS = {-1: 3,  # up
              1j: 0,  # right
              1: 1,  # down
              -1j: 2}  # left


class Cube:
    def __init__(self, data: list[list[str]], part2: bool = False) -> None:
        self.part2 = part2
        _map = [list(i) for i in data[0]]
        path = re.split(r'(\d+)([A-Z])', *data[1])
        self.path = iter(list(filter(None, path)))
        self.direction = 1j
        max_len = np.max([len(a) for a in _map])
        self.map = np.asarray(
            [np.pad(a, (0, max_len - len(a)), 'constant', constant_values=' ') for a in _map]
        )
        self.location = complex(0, np.nonzero(self.map[0] == '.')[0][0])

    def warp(self, loc: complex) -> complex:
        row, col = int(loc.real), int(loc.imag)
        match self.direction:
            case 1j: col = np.nonzero(self.map[row])[0][0]
            case -1j: col = np.nonzero(self.map[row])[0][-1]
            case -1: row = np.nonzero(self.map[:, col])[0][-1]
            case 1: row = np.nonzero(self.map[:, col])[0][0]
        return complex(row, col)

    def shift(self, loc: complex) -> tuple[complex, complex]:
        """Hardcoded rotations because screw this puzzle.

            |1|2|
            |3|
          |5|4|
          |6|

        """
        row, col = loc.real, loc.imag
        if 0 <= row <= 49 and 50 <= col <= 99: face = 1
        elif 0 <= row <= 49 and 100 <= col <= 149: face = 2
        elif 50 <= row <= 99 and 50 <= col <= 99: face = 3
        elif 100 <= row <= 149 and 50 <= col <= 99: face = 4
        elif 100 <= row <= 149 and 0 <= col <= 49: face = 5
        elif 150 <= row <= 199 and 0 <= col <= 49: face = 6
        else: raise IndexError('Point is not on any of the faces.')

        # positions relative to face
        rel_row, rel_col = row % 50, col % 50

        match face, self.direction:
            case 1, -1j: mv = complex(149 - rel_row, 0), 1j  # -> face 5
            case 1, -1: mv = complex(rel_col + 150, 0), 1j  # -> face 6

            case 2, -1: mv = complex(199, rel_col), -1  # -> face 6
            case 2, 1j: mv = complex(149 - rel_row, 99), -1j  # -> face 4
            case 2, 1: mv = complex(rel_col + 50, 99), -1j  # -> face 3

            case 3, -1j: mv = complex(100, rel_row), 1  # -> face 5
            case 3, 1j: mv = complex(49, rel_row + 100), -1  # -> face 2

            case 4, 1: mv = complex(rel_col + 150, 49), -1j  # -> face 6
            case 4, 1j: mv = complex(49 - rel_row, 149), -1j  # -> face 2

            case 5, -1j: mv = complex(49 - rel_row, 50), 1j  # -> face 1
            case 5, -1: mv = complex(rel_col + 50, 50), 1j  # -> face 3

            case 6, 1j: mv = complex(149, rel_row + 50), -1  # -> face 4
            case 6, 1: mv = complex(0, rel_col + 100), 1  # -> face 2
            case 6, -1j: mv = complex(0, rel_row + 50), 1  # -> face 1

            case _:
                raise NotImplementedError(f'No transition {self.direction} from face {face}.')
        return mv

    def move(self, distance: int) -> None:
        # print(distance, self.direction)
        for _ in range(distance):
            current_location = self.location
            current_direction = self.direction
            self.location += self.direction

            try:
                cell = self.map[int(self.location.real), int(self.location.imag)]
            except IndexError:
                cell = ' '

            match (cell, self.part2):
                case (' ', False): _next = self.warp(self.location)
                case (' ', True): _next, self.direction = self.shift(current_location)
                case _: _next = self.location

            match self.map[int(_next.real), int(_next.imag)]:
                case '.' | 'X': self.location = _next
                case '#': self.location, self.direction = current_location, current_direction

            self.map[int(self.location.real), int(self.location.imag)] = 'X'

    def start(self) -> int:
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
                self.direction *= -1j if turn == 'R' else 1j
            except StopIteration:
                self.location += 1+1j
                direction = DIRECTIONS[self.direction]
                # print(self.map)
                return int((1000 * self.location.real) + (4 * self.location.imag) + direction)


print(Cube(parsers.blocks(loader.get())).start())  # 76332
print(Cube(parsers.blocks(loader.get()), part2=True).start())  # 144012
