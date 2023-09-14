from numpy.typing import NDArray
from tools import parsers, loader
from tools.common import Point
import numpy as np


DIRECTIONS = {'U': Point(-1, 0), 'D': Point(1, 0), 'L': Point(0, -1), 'R': Point(0, 1)}
TEST = """ULL
RRDDD
LURDL
UUUUD
"""


def move(numpad: NDArray, data: list[str]) -> str:
    numpad = np.pad(numpad, 1, 'constant', constant_values=0)
    current = Point(*np.argwhere(numpad == '5')[0])
    code = ''
    for line in data:
        for i in line:
            _new = current + DIRECTIONS[i]
            if numpad[_new] != '0':
                current = _new
        code += numpad[current]
    return code


def part_1(data: list[str]) -> str:
    """
    >>> print(part_1(parsers.inline_test(TEST)))
    1985"""
    numpad = np.arange(1, 10).reshape((3, 3)).astype(str)
    return move(numpad, data)


def part_2(data: list[str]) -> str:
    """
    >>> print(part_2(parsers.inline_test(TEST)))
    5DB3"""
    numpad = np.asarray([[0, 0, 1, 0, 0],
                         [0, 2, 3, 4, 0],
                         [5, 6, 7, 8, 9],
                         [0, 'A', 'B', 'C', 0],
                         [0, 0, 'D', 0, 0]], dtype=str)
    return move(numpad, data)


print(part_1(parsers.lines(loader.get())))  # 65556
print(part_2(parsers.lines(loader.get())))  # CB779
