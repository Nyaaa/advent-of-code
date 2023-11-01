from itertools import combinations

import networkx as nx

from tools import loader, parsers


def count_constellations(data: list[str]) -> int:
    """
    >>> print(count_constellations(parsers.lines('test1.txt')))
    2
    >>> print(count_constellations(parsers.lines('test2.txt')))
    4
    >>> print(count_constellations(parsers.lines('test3.txt')))
    3
    >>> print(count_constellations(parsers.lines('test4.txt')))
    8"""
    points = [tuple(map(int, line.split(','))) for line in data]
    graph = nx.Graph()
    for a, b in combinations(points, 2):
        graph.add_nodes_from([a, b])
        if sum(abs(i - j) for i, j in zip(a, b)) <= 3:
            graph.add_edge(a, b)
    return nx.number_connected_components(graph)


print(count_constellations(parsers.lines(loader.get())))  # 327
