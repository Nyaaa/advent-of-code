import re
from typing import NamedTuple
from tools import parsers, loader
import numpy as np


class Claim(NamedTuple):
    id: int
    start_col: int
    start_row: int
    cols: int
    rows: int

    @property
    def sheet_size(self):
        return (slice(self.start_col, self.start_col + self.cols),
                slice(self.start_row, self.start_row + self.rows))


class Fabric:
    def __init__(self, data: list):
        self.array = np.zeros((1000, 1000), dtype=int)
        self.claims = [Claim(*map(int, re.findall(r'\d+', line))) for line in data]

    def part_1(self):
        """
        >>> print(Fabric(parsers.lines('test.txt')).part_1())
        4"""
        for claim in self.claims:
            self.array[claim.sheet_size] += 1
        return np.count_nonzero(self.array >= 2)

    def part_2(self):
        """
        >>> print(Fabric(parsers.lines('test.txt')).part_2())
        3"""
        self.part_1()
        for claim in self.claims:
            if np.all(self.array[claim.sheet_size] == 1):
                return claim.id


print(Fabric(parsers.lines(loader.get())).part_1())  # 118223
print(Fabric(parsers.lines(loader.get())).part_2())  # 412