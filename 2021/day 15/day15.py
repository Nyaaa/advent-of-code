from tools import parsers, loader
import networkx as nx
import numpy as np


class Cave:
    def __init__(self, data, part2: bool = False):
        self.grid = np.array([[int(i) for i in j] for j in (list(x) for x in data)])

        if part2:
            self.multiply_grid(0)
            self.multiply_grid(1)

        self.graph = nx.DiGraph()
        self.source = (0, 0)
        self.target = (self.grid.shape[0] - 1, self.grid.shape[1] - 1)
        self.build_graph()

    def multiply_grid(self, axis: int):
        grid = self.grid.copy()
        for _ in range(4):
            grid += 1
            grid[grid > 9] = 1
            self.grid = np.append(self.grid, grid, axis=axis)

    def build_graph(self):
        for index, item in np.ndenumerate(self.grid):
            row, col = index
            if row > 0:
                up = (row - 1, col)
                self.graph.add_edge(index, up, weight=self.grid[up])
            if row < (len(self.grid) - 1):
                down = (row + 1, col)
                self.graph.add_edge(index, down, weight=self.grid[down])
            if col > 0:
                left = (row, col - 1)
                self.graph.add_edge(index, left, weight=self.grid[left])
            if col < (len(self.grid[row]) - 1):
                right = (row, col + 1)
                self.graph.add_edge(index, right, weight=self.grid[right])

    def start(self):
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
