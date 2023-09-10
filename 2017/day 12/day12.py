import re
from tools import parsers, loader
import networkx as nx


class Pipes:
    def __init__(self, data: list[str]) -> None:
        self.G = nx.Graph()
        for line in data:
            name, *names = re.findall(r'\w+', line)
            for i in names:
                self.G.add_edge(name, i)

    def part_1(self) -> int:
        """
        >>> print(Pipes(parsers.lines('test.txt')).part_1())
        6"""
        return len(nx.single_source_shortest_path(self.G, '0'))

    def part_2(self) -> int:
        """
        >>> print(Pipes(parsers.lines('test.txt')).part_2())
        2"""
        return len(list(nx.connected_components(self.G)))


print(Pipes(parsers.lines(loader.get())).part_1())  # 175
print(Pipes(parsers.lines(loader.get())).part_2())  # 213
