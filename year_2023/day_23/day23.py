import networkx as nx
import numpy as np

from tools import common, loader, parsers


def trail(data: list[str]) -> int:
    """
    >>> print(trail(parsers.lines('test.txt')))
    94"""
    directions = {'>': (0, 1), '<': (0, -1), 'v': (1, 0), '^': (-1, 0)}
    arr = np.array([list(i) for i in data])
    start = tuple(np.argwhere(arr == '.')[0])
    finish = tuple(np.argwhere(arr == '.')[-1])
    graph = nx.DiGraph()
    for i, val in np.ndenumerate(arr):
        if val == '.':
            for adj_i, adj_val in common.get_adjacent(arr, i):
                if adj_val in '.^v<>':
                    graph.add_edge(i, adj_i)
                if adj_val == '.':
                    graph.add_edge(adj_i, i)
        elif val in '^v<>':
            d = directions[val]
            adj = i[0] + d[0], i[1] + d[1]
            graph.add_edge(i, adj)

    paths = nx.all_simple_paths(graph, start, finish)
    part1 = [len(i) - 1 for i in paths]
    return max(part1)


print(trail(parsers.lines(loader.get())))  # 1966
