import operator
from collections import defaultdict
from collections.abc import Generator
from itertools import combinations

from tools import loader, parsers
from tools.common import Point


def get_nodes_part1(a: Point, b: Point, d: Point, limit: int) -> Generator[Point]:
    for new_point in (a - d, a + d, b - d, b + d):
        a1 = a.manhattan_distance(new_point)
        b1 = b.manhattan_distance(new_point)
        if a1 and b1 and (b1 // a1 == 2 or a1 // b1 == 2) and new_point.is_on_grid(limit):
            yield new_point


def get_nodes_part2(a: Point, b: Point, d: Point, limit: int) -> Generator[Point]:
    for point in (a, b):
        for op in (operator.add, operator.sub):
            while True:
                point = op(point, d)
                if point.is_on_grid(limit):
                    yield point
                else:
                    break


def count_antinodes(data: list[str], part2: bool) -> int:
    """
    >>> print(count_antinodes(parsers.lines('test.txt'), part2=False))
    14
    >>> print(count_antinodes(parsers.lines('test2.txt'), part2=True))
    9
    >>> print(count_antinodes(parsers.lines('test.txt'), part2=True))
    34"""
    grid = defaultdict(set)
    limit = len(data)
    result = set()
    get_nodes = get_nodes_part2 if part2 else get_nodes_part1

    for row, line in enumerate(data):
        for col, char in enumerate(line):
            if char != '.':
                grid[char].add(Point(row, col))

    for points in grid.values():
        for a, b in combinations(points, 2):
            result.update(get_nodes(a, b, a - b, limit))
    return len(result)


print(count_antinodes(parsers.lines(loader.get()), part2=False))  # 398
print(count_antinodes(parsers.lines(loader.get()), part2=True))  # 1333
