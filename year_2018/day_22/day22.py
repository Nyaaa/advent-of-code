import re

import networkx as nx
import numpy as np

from tools import common, loader, parsers


class Cave:
    def __init__(self, data: str) -> None:
        self.depth, target_x, target_y = map(int, re.findall(r'\d+', data))
        self.target = (target_y, target_x)
        shape = (self.target[0] + 100, self.target[1] + 100)
        self.region_types = np.zeros(shape, dtype=int)

    def part_1(self) -> int:
        """
        >>> print(Cave('510, 10, 10').part_1())
        114"""
        geo = np.zeros_like(self.region_types)
        region_erosion = np.zeros_like(self.region_types)
        for (y, x) in np.ndindex(geo.shape):
            if y == 0:
                geo_index = x * 16807
            elif x == 0:
                geo_index = y * 48271
            else:
                geo_index = region_erosion[y, x - 1] * region_erosion[y - 1, x]
            if (y, x) in [(0, 0), self.target]:
                geo_index = 0
            erosion = (geo_index + self.depth) % 20183
            region_erosion[y, x] = erosion
            geo[y, x] = geo_index
            self.region_types[y, x] = erosion % 3
        return np.sum(self.region_types[:self.target[0] + 1, :self.target[1] + 1])

    def part_2(self) -> int:
        """
        tools: 0 = nothing, 1 = torch, 2 = gear
        terrain: 0 = rocky, 1 = wet, 2 = narrow

        >>> print(Cave('510, 10, 10').part_2())
        45"""
        self.part_1()
        tools = {0: (2, 1), 1: (2, 0), 2: (1, 0)}
        graph = nx.Graph()
        for curr_loc, curr_terrain in np.ndenumerate(self.region_types):
            cur_tools = tools[curr_terrain]
            graph.add_edge(
                (curr_loc, cur_tools[0]), (curr_loc, cur_tools[1]), weight=7)
            for adj_loc, adj_terrain in common.get_adjacent(self.region_types, curr_loc):
                for t in set(cur_tools).intersection(tools[adj_terrain]):
                    graph.add_edge((curr_loc, t), (adj_loc, t), weight=1)
        return nx.dijkstra_path_length(graph, ((0, 0), 1), (self.target, 1))


print(Cave(parsers.string(loader.get())).part_1())  # 11972
print(Cave(parsers.string(loader.get())).part_2())  # 1092
