import re
from math import prod

import networkx as nx

from tools import loader, parsers


def snow_machine(data: list[str]) -> int:
    """
    >>> print(snow_machine(parsers.lines('test.txt')))
    54"""
    graph = nx.Graph()
    for line in data:
        source, *nodes = re.findall(r'\w+', line)
        graph.add_edges_from([(source, node) for node in nodes])
    graph.remove_edges_from(nx.minimum_edge_cut(graph))
    return prod(len(c) for c in nx.connected_components(graph))


print(snow_machine(parsers.lines(loader.get())))  # 520380
