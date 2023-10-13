from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from tools import common, loader, parsers
from year_2019 import intcode

DIRECTIONS = {'up': ('left', 'right'),
              'right': ('up', 'down'),
              'down': ('right', 'left'),
              'left': ('down', 'up')}
MOVES = {'up': (1, 0), 'down': (-1, 0),
         'left': (0, -1), 'right': (0, 1)}
np.set_printoptions(linewidth=np.inf)


class Robot:
    def __init__(self) -> None:
        self.pc = intcode.Intcode(parsers.lines(loader.get()))
        self.path = []
        self.direction = 'up'
        self.current_loc = (50, 50)
        # Hardcoded array size, may not be enough for some cases.
        self.grid = np.zeros(shape=(150, 150), dtype=int)

    def paint(self) -> None:
        while True:
            self.path.append(self.current_loc)
            move = self.pc.run([self.grid[self.current_loc]])
            if self.pc.done:
                break
            colour, turn = move
            if self.grid[self.current_loc] != colour:
                self.grid[self.current_loc] = colour
            self.direction = DIRECTIONS[self.direction][turn]
            self.current_loc = (self.current_loc[0] + MOVES[self.direction][0],
                                self.current_loc[1] + MOVES[self.direction][1])

    def part_1(self) -> int:
        self.paint()
        return len(set(self.path))

    def part_2(self) -> NDArray:
        self.grid[self.current_loc] = 1
        self.paint()
        trimmed = common.trim_array(self.grid)
        image = common.convert_to_image(trimmed)
        return np.flipud(image)


print(Robot().part_1())  # 2883
print(Robot().part_2())  # LEPCPLGZ
