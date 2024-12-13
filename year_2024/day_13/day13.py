import re
from fractions import Fraction
from itertools import starmap

import numpy as np

from tools import loader, parsers


def numpy_solve(ints: list[int]) -> int:
    buttons = np.array([[ints[0], ints[2]], [ints[1], ints[3]]])
    prize = np.array([ints[4], ints[5]])
    solution = np.linalg.solve(buttons, prize)
    if all(np.abs(np.round(solution) - solution) < 0.01):
        return int(np.round(solution[0] * 3 + solution[1]))
    return 0


def solve(ax: int, ay: int, bx: int, by: int, x: int, y: int) -> int:  # noqa: PLR0913, PLR0917
    """
    ax * i + bx * j = x
    ay * i + by * j = y

    ax * i = x - bx * j
    i = (x - bx * j) / ax

    ay * ((x - bx * j) / ax) = y - by * j
    ay * (x - bx * j) = ax * (y - by * j)
    ay * x - ay * (bx * j) = ax * y - (ax * by * j)
    ay * x + (ay * bx - ax * by) * j = ax * y
    (ay * bx - ax * by) * j = ax * y - ay * x
    j = (ax * y - ay * x) / (ay * bx - ax * by)
    """
    j = Fraction(abs((ax * y - ay * x) / (ay * bx - ax * by)))
    i = Fraction((x - bx * j) / ax)
    if i.denominator == 1 and j.denominator == 1:
        return i.numerator * 3 + j.numerator
    return 0


def play_arcade(data: list[list[str]], multiplier: int = 0) -> int:
    """
    >>> print(play_arcade(parsers.blocks('test.txt')))
    480"""
    machines = []
    for block in data:
        ints = [int(i) for line in block for i in re.findall(r'\d+', line)]
        ints[-1] += multiplier
        ints[-2] += multiplier
        machines.append(ints)
    # return sum(starmap(numpy_solve, machines))
    return sum(starmap(solve, machines))


print(play_arcade(parsers.blocks(loader.get())))  # 31552
print(play_arcade(parsers.blocks(loader.get()), multiplier=10000000000000))  # 95273925552482
