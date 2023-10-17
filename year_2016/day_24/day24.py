from functools import cache
from itertools import pairwise, permutations

import networkx as nx
import numpy as np

from tools import common, loader, parsers


@cache
def get_distance(graph: nx.Graph, a: tuple[int], b: tuple[int]) -> int:
    return nx.shortest_path_length(graph, a, b)


def maze(data: list[str], part2: bool) -> int:
    """
    >>> print(maze(parsers.lines('test.txt'), False))
    14"""
    graph = nx.Graph()
    grid = np.asarray([list(line) for line in data], dtype=str)
    poi = {}
    for loc, val in np.ndenumerate(grid):
        if val != '#':
            for _loc, v in common.get_adjacent(grid, loc):
                if v != '#':
                    graph.add_edge(_loc, loc)
        if val.isnumeric():
            graph.nodes[loc]['label'] = int(val)
            poi[val] = loc
    start = poi.pop('0')

    shortest = np.inf
    for p in permutations(poi.values(), len(poi)):
        length = 0
        path = [start, *p]
        if part2:
            path.append(start)
        for a, b in pairwise(path):
            length += get_distance(graph, a, b)
        if length < shortest:
            shortest = length
    return shortest


print(maze(parsers.lines(loader.get()), False))  # 490
print(maze(parsers.lines(loader.get()), True))  # 744
