from itertools import permutations

import networkx as nx
from networkx.classes.function import path_weight

from tools import loader, parsers


def salesman(data: list[str]) -> tuple[int, int]:
    """
    >>> print(salesman(parsers.lines('test.txt')))
    (605, 982)"""
    graph = nx.Graph()
    for i in data:
        vals = i.split()
        graph.add_edge(vals[0], vals[2], weight=int(vals[4]))
    lengths = {path_weight(graph, path, weight='weight')
               for path in permutations(graph.nodes, graph.number_of_nodes())}
    return min(lengths), max(lengths)


print(salesman(parsers.lines(loader.get())))  # 207, 804
