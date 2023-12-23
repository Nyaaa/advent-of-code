import networkx as nx
import numpy as np
from networkx import path_weight

from tools import common, loader, parsers


def trail(data: list[str], part2: bool) -> int:
    """
    >>> print(trail(parsers.lines('test.txt'), part2=False))
    94
    >>> print(trail(parsers.lines('test.txt'), part2=True))
    154
    """
    directions = {'>': (0, 1), '<': (0, -1), 'v': (1, 0), '^': (-1, 0)}
    arr = np.array([list(i) for i in data])
    start = tuple(np.argwhere(arr == '.')[0])
    finish = tuple(np.argwhere(arr == '.')[-1])
    graph = nx.DiGraph()
    for i, val in np.ndenumerate(arr):
        if val == '.':
            for adj_i, adj_val in common.get_adjacent(arr, i):
                if adj_val in '.^v<>':
                    graph.add_edge(i, adj_i, weight=1)
                if adj_val == '.':
                    graph.add_edge(adj_i, i, weight=1)
        elif val in '^v<>':
            d = directions[val]
            graph.add_edge(i, (i[0] + d[0], i[1] + d[1]), weight=1)

    if part2:
        graph = graph.to_undirected(as_view=False)
        straight = [n for n in graph.nodes if len(graph.edges(n)) == 2]
        for node in straight:
            adj1, adj2 = graph.neighbors(node)
            weight = path_weight(graph, [adj1, node, adj2], 'weight')
            graph.add_edge(adj1, adj2, weight=weight)
            graph.remove_node(node)

    paths = nx.all_simple_paths(graph, start, finish)
    return max(path_weight(graph, path, 'weight') for path in paths)


print(trail(parsers.lines(loader.get()), part2=False))  # 1966
print(trail(parsers.lines(loader.get()), part2=True))  # 6286
