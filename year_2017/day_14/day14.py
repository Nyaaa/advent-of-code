from importlib.machinery import SourceFileLoader

import numpy as np

from tools import common, loader, parsers

hasher = SourceFileLoader('knot_hash', '../day_10/day10.py').load_module()


class Defragment:
    def __init__(self, data: str) -> None:
        self.grid = np.empty((128, 128), dtype=int)
        for i in range(128):
            _hash = hasher.part_2(f'{data}-{i}')
            _bin = np.binary_repr(int(_hash, 16), width=128)
            self.grid[i] = list(_bin)

    def check_adjacent(self, cell: tuple[int, int]) -> int:
        self.grid[cell] = 0
        for adj_i, adj_val in common.get_adjacent(self.grid, cell):
            if adj_val:
                self.check_adjacent(adj_i)
        return 1

    def part_1(self) -> int:
        """
        >>> print(Defragment('flqrgnkx').part_1())
        8108"""
        return np.count_nonzero(self.grid)

    def part_2(self) -> int:
        """
        >>> print(Defragment('flqrgnkx').part_2())
        1242"""
        out = 0
        for i, val in np.ndenumerate(self.grid):
            if val:
                out += self.check_adjacent(i)
        return out


d = Defragment(parsers.string(loader.get()))
print(d.part_1())  # 8106
print(d.part_2())  # 1164
