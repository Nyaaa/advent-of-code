from tools import parsers, loader
import string
import networkx as nx
import numpy as np


test = """Sabqponm
abcryxxl
accszExk
acctuvwj
abdefghi
"""


class Graph:
    letters = list(string.ascii_lowercase)
    numbers = list(range(26))
    weights = dict(zip(letters, numbers))
    weights['S'] = 0
    weights['E'] = 25

    def __init__(self, array: list, start: str):
        _grid = []
        self.start_coord = []
        self.end_coord = None
        self.start_letter = start
        self.end_letter = 'E'
        self.digraph = nx.DiGraph()

        for row, line in enumerate(array):
            new_row = []
            for col, letter in enumerate(line):
                if letter == self.start_letter:
                    self.start_coord.append((row, col))
                elif letter == self.end_letter:
                    self.end_coord = (row, col)
                new_row.append(letter)
            _grid.append(new_row)

        self.grid = np.array(_grid, dtype=str)
        self.build_graph()

    def build_graph(self):
        for index, item in np.ndenumerate(self.grid):
            row, col = index
            if row > 0:
                up = self.grid[row - 1][col]
                self.add_edges(up, index, item, (row - 1, col))

            if row < (len(self.grid) - 1):
                down = self.grid[row + 1][col]
                self.add_edges(down, index, item, (row + 1, col))

            if col > 0:
                left = self.grid[row][col - 1]
                self.add_edges(left, index, item, (row, col - 1))

            if col < (len(self.grid[row]) - 1):
                right = self.grid[row][col + 1]
                self.add_edges(right, index, item, (row, col + 1))

    def add_edges(self, char: str, index: tuple[int, int], letter: str, v: tuple[int, int]):
        weight = self.weights[letter]
        if self.weights[char] <= weight + 1:
            self.digraph.add_edge(index, v)

    def solve(self):
        """test part1:
        >>> Graph(parsers.inline_test(test), 'S').solve()
        31

        test part 2:
        >>> Graph(parsers.inline_test(test), 'a').solve()
        29
        """
        distance = float('inf')
        for point in self.start_coord:
            try:
                attempt = nx.shortest_path_length(self.digraph, source=point, target=self.end_coord)
            except nx.exception.NetworkXNoPath:
                continue
            if attempt <= distance:
                distance = attempt
        return distance


# part 1
print(Graph(parsers.lines(loader.get()), 'S').solve())  # 449
# part 2
print(Graph(parsers.lines(loader.get()), 'a').solve())  # 443
