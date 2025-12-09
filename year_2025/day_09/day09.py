from __future__ import annotations

from itertools import combinations, starmap

from shapely import Polygon
from shapely.geometry import box

from tools import loader, parsers


def get_area(a: tuple, b: tuple) -> int:
    return (abs(a[0] - b[0]) + 1) * (abs(a[1] - b[1]) + 1)


def part_1(data: list[str]) -> int:
    """
    >>> print(part_1(parsers.lines('test.txt')))
    50
    """
    points = [tuple(map(int, line.split(','))) for line in data]
    return max(starmap(get_area, combinations(points, 2)))


def part_2(data: list[str]) -> int:
    """
    >>> print(part_2(parsers.lines('test.txt')))
    24
    """
    points = [tuple(map(int, line.split(','))) for line in data]
    polygon = Polygon(points)

    def get_inner_area(a: tuple, b: tuple) -> int:
        inner = box(min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))
        if polygon.contains(inner):
            return get_area(a, b)
        return 0

    return max(starmap(get_inner_area, combinations(points, 2)))


print(part_1(parsers.lines(loader.get())))  # 4764078684
print(part_2(parsers.lines(loader.get())))  # 1652344888
