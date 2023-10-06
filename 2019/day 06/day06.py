import networkx as nx

from tools import loader, parsers


class Orbits:
    def __init__(self, data: list) -> None:
        self.planets = nx.Graph()
        for line in data:
            planet1, planet2 = line.split(')')
            self.planets.add_edge(planet1, planet2)

    def part_1(self) -> int:
        """
        >>> print(Orbits(parsers.lines('test.txt')).part_1())
        42"""
        return sum(nx.dijkstra_path_length(self.planets, 'COM', p) for p in self.planets.nodes)

    def part_2(self) -> int:
        """
        >>> print(Orbits(parsers.lines('test2.txt')).part_2())
        4"""
        return nx.dijkstra_path_length(self.planets, 'YOU', 'SAN') - 2


print(Orbits(parsers.lines(loader.get())).part_1())  # 186597
print(Orbits(parsers.lines(loader.get())).part_2())  # 412
