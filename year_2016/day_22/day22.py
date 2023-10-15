from __future__ import annotations

import re
from itertools import permutations
from typing import NamedTuple

import numpy as np

from tools import loader, parsers


class Node(NamedTuple):
    x: int
    y: int
    size: int
    used: int
    avail: int

    def __eq__(self, other: Node) -> bool:
        return self.x == other.x and self.y == other.y

    def is_viable(self, other: Node) -> bool:
        return self.used != 0 and self != other and self.used <= other.avail


class Puzzle:
    def __init__(self, data: list[str]) -> None:
        self.nodes = [Node(*map(int, n[:-1]))
                      for n in (re.findall(r'\d+', node) for node in data) if n]

    def part_1(self) -> int:
        return sum(a.is_viable(b) for a, b in permutations(self.nodes, 2))

    def part_2(self) -> int:
        nodes = sorted(self.nodes, key=lambda i: (i.x, i.y))
        grid = np.zeros(shape=(nodes[-1].y + 1, nodes[-1].x + 1), dtype=int)
        for n in nodes:
            if n.size > 100:
                grid[n.y, n.x] = 1
            elif n.used == 0:
                grid[n.y, n.x] = 2
        # np.set_printoptions(threshold=2000)
        # print(grid)
        # 3 left, 28 up, 31 right, 5*31, 1 right
        return 3 + 28 + 31 + 5 * 31 + 1


print(Puzzle(parsers.lines(loader.get())).part_1())  # 990
print(Puzzle(parsers.lines(loader.get())).part_2())  # 218
