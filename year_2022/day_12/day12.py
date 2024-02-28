from pathlib import Path

import networkx as nx
import numpy as np

from tools import common, loader


class Graph:
    def __init__(self, data: str | Path, start: str) -> None:
        grid = np.genfromtxt(data, delimiter=1, dtype=str)
        self.start_coord = np.argwhere(grid == start)
        self.end_coord = tuple(np.argwhere(grid == 'E')[0])
        self.digraph = nx.DiGraph()
        grid[grid == 'E'] = 'z'
        grid[grid == 'S'] = 'a'
        numerical = grid.view(np.int32)

        for index, letter in np.ndenumerate(numerical):
            for ajd_index, adj_value in common.get_adjacent(numerical, index):
                if adj_value <= letter + 1:
                    self.digraph.add_edge(index, ajd_index)

    def solve(self) -> int:
        """
        >>> Graph('test.txt', 'S').solve()
        31
        >>> Graph('test.txt', 'a').solve()
        29
        """
        distance = float('inf')
        for point in self.start_coord:
            try:
                attempt = nx.shortest_path_length(
                    self.digraph, source=tuple(point), target=self.end_coord
                )
            except nx.exception.NetworkXNoPath:
                continue
            distance = min(attempt, distance)
        return distance


print(Graph(loader.get(), 'S').solve())  # 449
print(Graph(loader.get(), 'a').solve())  # 443
