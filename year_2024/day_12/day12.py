import heapq
from collections import defaultdict
from itertools import starmap

import networkx as nx
import numpy as np

from tools import common, loader, parsers
from tools.common import Point


def count_sides(group: set[tuple[int, int]]) -> int:
    outer_nodes = list(starmap(Point, group))
    edges = defaultdict(list)
    nodes = set(outer_nodes)
    for node in outer_nodes:
        if (new := node + Point(-1, 0)) not in nodes:
            edges[f'above {node.row}'].append(new.col)
        if (new := node + Point(1, 0)) not in nodes:
            edges[f'below {node.row}'].append(new.col)
        if (new := node + Point(0, -1)) not in nodes:
            edges[f'left {node.col}'].append(new.row)
        if (new := node + Point(0, 1)) not in nodes:
            edges[f'right {node.col}'].append(new.row)

    edge_count = 0
    for values in edges.values():
        heapq.heapify(values)
        edge_count += 1
        previous = heapq.heappop(values)
        while values:
            current = heapq.heappop(values)
            if current - previous != 1:
                edge_count += 1
            previous = current

    return edge_count


def get_fence(data: list[str], part2: bool) -> int:
    """
    >>> print(get_fence(parsers.lines('test3.txt'), part2=False))
    1930
    >>> print(get_fence(parsers.lines('test.txt'), part2=True))
    80
    >>> print(get_fence(parsers.lines('test2.txt'), part2=True))
    236
    >>> print(get_fence(parsers.lines('test4.txt'), part2=True))
    368
    """
    arr = np.genfromtxt(data, delimiter=1, dtype=str)
    graph = nx.Graph()
    for i, val in np.ndenumerate(arr):
        graph.add_node(i)
        for adj_i, adj_val in common.get_adjacent(arr, i):
            if val == adj_val:
                graph.add_edge(i, adj_i)

    result = 0
    perimeter_counts = {2: 2, 3: 1, 1: 3, 0: 4, 4: 0}  # number of edges -> number of fence tiles
    for group in nx.connected_components(graph):
        area = len(group)
        if not part2:
            result += area * sum(perimeter_counts[graph.degree(node)] for node in group)
        else:
            result += area * count_sides(group)

    return result


print(get_fence(parsers.lines(loader.get()), part2=False))  # 1533644
print(get_fence(parsers.lines(loader.get()), part2=True))  # 936718
