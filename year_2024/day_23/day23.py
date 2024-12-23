import networkx as nx

from tools import loader, parsers


def inspect_lan(data: list[str]) -> list[list[str]]:
    graph = nx.Graph()
    for line in data:
        node_a, node_b = line.split('-')
        graph.add_edge(node_a, node_b)
    return list(nx.enumerate_all_cliques(graph))


def part_1(data: list[str]) -> int:
    """
    >>> print(part_1(parsers.lines('test.txt')))
    7"""
    cliques = inspect_lan(data)
    return len([i for i in cliques if any(j.startswith('t') for j in i) and len(i) == 3])


def part_2(data: list[str]) -> str:
    """
    >>> print(part_2(parsers.lines('test.txt')))
    co,de,ka,ta"""
    cliques = inspect_lan(data)
    return ','.join(sorted(max(cliques, key=len)))


print(part_1(parsers.lines(loader.get())))  # 1269
print(part_2(parsers.lines(loader.get())))  # ad,jw,kt,kz,mt,nc,nr,sb,so,tg,vs,wh,yh
