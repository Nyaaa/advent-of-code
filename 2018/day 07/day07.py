import networkx as nx
from tools import parsers, loader


class Sleigh:
    def __init__(self, data: list[str]) -> None:
        edges = [(j[1], j[7]) for j in (i.split() for i in data)]
        self.G = nx.DiGraph(edges)

    def part_1(self) -> str:
        """
        >>> print(Sleigh(parsers.lines('test.txt')).part_1())
        CABDFE"""
        return ''.join(nx.lexicographical_topological_sort(self.G))

    def part_2(self) -> int:
        time = 0
        for node in self.G.nodes:
            self.G.nodes[node]['work'] = ord(node) - 4

        while self.G:
            work = [i for i, j in self.G.in_degree if j == 0]
            for task in work[:5]:
                self.G.nodes[task]['work'] -= 1
                if self.G.nodes[task]['work'] == 0:
                    self.G.remove_node(task)
            time += 1
        return time


print(Sleigh(parsers.lines(loader.get())).part_1())  # BHRTWCYSELPUVZAOIJKGMFQDXN
print(Sleigh(parsers.lines(loader.get())).part_2())  # 959
