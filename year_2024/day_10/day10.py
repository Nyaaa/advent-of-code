import networkx as nx
import numpy as np

from tools import common, loader, parsers


def chart_trail(data: list[str], part2: bool) -> int:
    """
    >>> print(chart_trail(parsers.lines('test.txt'), part2=False))
    36
    >>> print(chart_trail(parsers.lines('test.txt'), part2=True))
    81"""
    arr = np.genfromtxt(data, delimiter=1, dtype=int)
    starts = [tuple(i) for i in np.argwhere(arr == 0)]
    finishes = [tuple(i) for i in np.argwhere(arr == 9)]
    graph = nx.DiGraph()
    for i, val in np.ndenumerate(arr):
        for adj_i, adj_val in common.get_adjacent(arr, i):
            if adj_val - val == 1:
                graph.add_edge(i, adj_i)

    count = 0
    for start in starts:
        if not part2:
            for finish in finishes:
                count += nx.has_path(graph, source=start, target=finish)
        else:
            count += len(list(nx.all_simple_paths(graph, source=start, target=finishes)))

    return count


print(chart_trail(parsers.lines(loader.get()), part2=False))  # 744
print(chart_trail(parsers.lines(loader.get()), part2=True))  # 1651
