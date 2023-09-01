from numpy.typing import NDArray
from tools import parsers, loader, common
import numpy as np


class Bugs:
    def __init__(self, data: list[str]):
        self.arr = np.array([list(1 if i == '#' else 0 for i in row) for row in data],
                            dtype=np.dtype('u1'))
        self.levels: dict[int, NDArray] = dict()
        self.size = self.arr.shape[0]
        mid = self.size // 2
        self.middle = (mid, mid)  # assuming square shape
        self.outer_slices = {-1: np.s_[mid - 1], self.size: np.s_[mid + 1]}
        self.inner_slices = {(mid - 1, mid): np.s_[0],
                             (mid, mid + 1): np.s_[:, -1],
                             (mid + 1, mid): np.s_[-1],
                             (mid, mid - 1): np.s_[:, 0]}

    def part_1(self) -> int:
        """
        >>> print(Bugs(parsers.lines('test.txt')).part_1())
        2129920"""
        previous = set()
        _arr = None
        while True:
            _arr = self.arr.copy()
            previous.add(hash(_arr.tobytes()))
            for i, val in np.ndenumerate(self.arr):
                adj = sum(i for _, i in common.get_adjacent(self.arr, i))
                if val and adj != 1:
                    _arr[i] = 0
                elif not val and adj == 1 or adj == 2:
                    _arr[i] = 1
            if hash(_arr.tobytes()) in previous:
                break
            self.arr = _arr
        return sum(2 ** i for i in np.flatnonzero(_arr))

    def count_adjacent(self, level: int, pos: tuple) -> int:
        count = 0
        if self.levels.get(level + 1) is None or self.levels.get(level - 1) is None:
            return 0
        for j in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            adj = (pos[0] + j[0], pos[1] + j[1])
            if 0 <= adj[0] < self.size and 0 <= adj[1] < self.size:
                if adj == self.middle:
                    count += np.count_nonzero(self.levels[level + 1][self.inner_slices[pos]])
                else:
                    count += self.levels[level][adj]
            else:
                r = self.outer_slices.get(adj[0], 2)
                c = self.outer_slices.get(adj[1], 2)
                count += self.levels[level - 1][r][c]
        return count

    def part_2(self, minutes: int) -> int:
        """
        >>> print(Bugs(parsers.lines('test.txt')).part_2(10))
        99"""
        depth = minutes // 2
        self.levels = {i: np.zeros_like(self.arr) for i in range(-depth - 1, depth + 2)}
        self.levels[0] = self.arr
        for _ in range(minutes):
            _levels = {}
            for level, arr in self.levels.items():
                _arr = np.zeros_like(self.arr)
                for i in np.ndindex(_arr.shape):
                    if i != self.middle:
                        adj = self.count_adjacent(level, i)
                        if (arr[i] and adj == 1) or (not arr[i] and (adj == 1 or adj == 2)):
                            _arr[i] = 1
                _levels[level] = _arr
            self.levels = _levels
        return sum(np.count_nonzero(i) for i in self.levels.values())


print(Bugs(parsers.lines(loader.get())).part_1())  # 18842609
print(Bugs(parsers.lines(loader.get())).part_2(200))  # 2059
