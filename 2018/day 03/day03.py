import re
from typing import NamedTuple

import numpy as np

from tools import loader, parsers


class Claim(NamedTuple):
    ident: int
    start_col: int
    start_row: int
    cols: int
    rows: int

    @property
    def sheet_size(self) -> tuple[slice, slice]:
        return (slice(self.start_col, self.start_col + self.cols),
                slice(self.start_row, self.start_row + self.rows))


class Fabric:
    def __init__(self, data: list[str]) -> None:
        self.array = np.zeros((1000, 1000), dtype=int)
        self.claims = [Claim(*map(int, re.findall(r'\d+', line))) for line in data]

    def part_1(self) -> int:
        """
        >>> print(Fabric(parsers.lines('test.txt')).part_1())
        4"""
        for claim in self.claims:
            self.array[claim.sheet_size] += 1
        return np.count_nonzero(self.array >= 2)

    def part_2(self) -> int:
        """
        >>> print(Fabric(parsers.lines('test.txt')).part_2())
        3"""
        self.part_1()
        for claim in self.claims:
            if np.all(self.array[claim.sheet_size] == 1):
                return claim.ident
        return None


print(Fabric(parsers.lines(loader.get())).part_1())  # 118223
print(Fabric(parsers.lines(loader.get())).part_2())  # 412
