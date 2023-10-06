import networkx as nx
import numpy as np

from tools import common, loader, parsers


class Cave:
    def __init__(self, data: list[str], part2: bool = False) -> None:
        self.grid = np.array([[int(i) for i in j] for j in (list(x) for x in data)])

        if part2:
            self.multiply_grid(0)
            self.multiply_grid(1)

        self.graph = nx.DiGraph()
        self.source = (0, 0)
        self.target = (self.grid.shape[0] - 1, self.grid.shape[1] - 1)
        self.build_graph()

    def multiply_grid(self, axis: int) -> None:
        grid = self.grid.copy()
        for _ in range(4):
            grid += 1
            grid[grid > 9] = 1
            self.grid = np.append(self.grid, grid, axis=axis)

    def build_graph(self) -> None:
        for index in np.ndindex(self.grid.shape):
            for ajd_index, value in common.get_adjacent(self.grid, index):
                self.graph.add_edge(index, ajd_index, weight=value)

    def start(self) -> int:
        """test part 1:
        >>> print(Cave(parsers.lines('test.txt')).start())
        40

        test part 2:
        >>> print(Cave(parsers.lines('test.txt'), part2=True).start())
        315"""

        path = nx.astar_path(self.graph, source=self.source, target=self.target, weight='weight')
        return nx.path_weight(self.graph, path, weight='weight')


print(Cave(parsers.lines(loader.get()), part2=False).start())  # 589
print(Cave(parsers.lines(loader.get()), part2=True).start())  # 2885
