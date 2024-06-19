from collections import defaultdict
from string import ascii_uppercase

import networkx as nx
import numpy as np

from tools import common, loader, parsers


class Maze:
    def __init__(self, data: list[str]) -> None:
        self.map = np.asarray([list(i.strip('\n')) for i in data], dtype=str)  # CRLF inputs
        self.G: nx.Graph = nx.grid_graph([self.map.shape[1], self.map.shape[0]])
        self.G.remove_nodes_from([tuple(i) for i in np.argwhere(self.map != '.')])
        self.portals = defaultdict(list)
        for k, v in self.find_portals().items():
            self.portals[v].append(k)
        self.start = self.portals.pop('AA')[0]
        self.stop = self.portals.pop('ZZ')[0]

    def find_portals(self) -> dict[tuple[int, int], str]:
        portals = {}
        for i, val in np.ndenumerate(self.map):
            if val in ascii_uppercase:
                label = val
                location = None
                adj_letter = None
                for loc, v in common.get_adjacent(self.map, i):
                    if v in ascii_uppercase:
                        label += v
                        adj_letter = loc
                    elif v == '.':
                        location = loc
                if not location:
                    try:
                        location = next(
                            i for i, j in common.get_adjacent(self.map, adj_letter) if j == '.'
                        )
                    except StopIteration:
                        continue
                if location not in portals:
                    portals[location] = label
        return portals

    def part_1(self) -> int:
        """
        >>> print(Maze(parsers.lines('test.txt', strip=False)).part_1())
        23

        >>> print(Maze(parsers.lines('test2.txt', strip=False)).part_1())
        58"""
        self.G.add_edges_from(self.portals.values())
        return nx.shortest_path_length(self.G, self.start, self.stop)

    def part_2(self) -> int:
        """
        >>> print(Maze(parsers.lines('test3.txt', strip=False)).part_2())
        396"""
        new_g = nx.Graph()
        for i in range(len(self.portals)):  # may need to increase recursion depth
            for u, v in self.G.edges:
                new_g.add_edge((*u, i), (*v, i))
            rows, cols = self.map.shape
            for tp in self.portals.values():
                if tp[0][0] in {2, rows - 3} or tp[0][1] in {2, cols - 3}:
                    outer, inner = tp
                else:
                    inner, outer = tp
                new_g.add_edge((*inner, i), (*outer, i + 1))
        return nx.shortest_path_length(new_g, (*self.start, 0), (*self.stop, 0))


print(Maze(parsers.lines(loader.get(), strip=False)).part_1())  # 490
print(Maze(parsers.lines(loader.get(), strip=False)).part_2())  # 5648
