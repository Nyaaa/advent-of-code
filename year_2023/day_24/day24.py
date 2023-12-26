from __future__ import annotations

import re
from itertools import combinations
from typing import NamedTuple

import sympy as sym

from tools import loader, parsers


class Hailstone(NamedTuple):
    px: int
    py: int
    pz: int
    vx: int
    vy: int
    vz: int

    @property
    def slope(self) -> float:
        return self.vy / self.vx

    @property
    def intercept(self) -> float:
        return self.py - self.px * self.slope

    def get_intersection(self, other: Hailstone) -> tuple[float, float]:
        if self.slope == other.slope:
            raise ValueError('Hailstones are parallel')
        x = (other.intercept - self.intercept) / (self.slope - other.slope)
        y = self.slope * x + self.intercept
        return x, y


def part_1(data: list[str], low_lim: int, high_lim: int) -> int:
    """
    >>> print(part_1(parsers.lines('test.txt'), low_lim=7, high_lim=27))
    2"""
    hailstones = [Hailstone(*map(int, re.findall(r'-?\d+', line))) for line in data]

    result = 0
    for a, b in combinations(hailstones, 2):
        try:
            x, y = a.get_intersection(b)
        except ValueError:
            continue
        time1 = (x - a.px) / a.vx
        time2 = (x - b.px) / b.vx
        if low_lim <= x <= high_lim and low_lim <= y <= high_lim and time1 >= 0 and time2 >= 0:
            result += 1
    return result


def part_2(data: list[str]) -> int:
    """
    >>> print(part_2(parsers.lines('test.txt')))
    47"""
    hailstones = [tuple(map(int, re.findall(r'-?\d+', line))) for line in data]
    equations = []
    for i, (px, py, pz, vx, vy, vz) in enumerate(hailstones[:3]):
        v = sym.var(f't{i}')
        equations.extend([
            sym.Eq(px + vx * v, sym.S(f'px + vx * {v}')),
            sym.Eq(py + vy * v, sym.S(f'py + vy * {v}')),
            sym.Eq(pz + vz * v, sym.S(f'pz + vz * {v}')),
        ])
    result = sym.solve(equations, dict=True)[0]
    return result[sym.var('px')] + result[sym.var('py')] + result[sym.var('pz')]


print(part_1(parsers.lines(loader.get()),
             low_lim=200000000000000,
             high_lim=400000000000000))  # 13910
print(part_2(parsers.lines(loader.get())))  # 618534564836937
