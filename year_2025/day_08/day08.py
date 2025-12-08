from __future__ import annotations

import math
from itertools import combinations, islice
from typing import TYPE_CHECKING

import networkx as nx

from tools import loader, parsers

if TYPE_CHECKING:
    from collections.abc import Generator


class Point:
    def __init__(self, *args: int) -> None:
        self.x, self.y, self.z = args

    def __repr__(self) -> str:
        return f'Point({self.x}, {self.y}, {self.z})'

    def get_distance(self, other: Point) -> float:
        return ((self.x - other.x) ** 2) + ((self.y - other.y) ** 2) + ((self.z - other.z) ** 2)


def get_sorted_points(points: list[Point]) -> Generator[tuple[Point, Point]]:
    combos = sorted(combinations(points, 2), key=lambda x: x[0].get_distance(x[1]))
    yield from combos


def part_1(data: list[str], connections: int = 1000) -> int:
    """
    >>> print(part_1(parsers.lines('test.txt'), connections=10))
    40
    """
    points = [Point(*map(int, line.split(','))) for line in data]
    distances = get_sorted_points(points)
    graph = nx.Graph()
    graph.add_edges_from(islice(distances, connections))
    components = sorted((len(i) for i in nx.connected_components(graph)), reverse=True)
    return math.prod(components[:3])


def part_2(data: list[str]) -> int:
    """
    >>> print(part_2(parsers.lines('test.txt')))
    25272
    """
    points = [Point(*map(int, line.split(','))) for line in data]
    distances = get_sorted_points(points)
    graph = nx.Graph()
    graph.add_nodes_from(points)
    point_1 = point_2 = None
    while not nx.is_connected(graph):
        point_1, point_2 = next(distances)
        graph.add_edge(point_1, point_2)
    return point_1.x * point_2.x


print(part_1(parsers.lines(loader.get())))  # 171503
print(part_2(parsers.lines(loader.get())))  # 9069509600
