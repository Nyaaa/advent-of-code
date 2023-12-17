import string

import networkx as nx
import numpy as np

from tools import common, loader, parsers

test = """Sabqponm
abcryxxl
accszExk
acctuvwj
abdefghi
"""
LETTERS = tuple(string.ascii_lowercase)
NUMBERS = tuple(range(26))
WEIGHTS = dict(zip(LETTERS, NUMBERS, strict=True))
WEIGHTS['S'] = 0
WEIGHTS['E'] = 25


class Graph:
    def __init__(self, array: list, start: str) -> None:
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

    def build_graph(self) -> None:
        for index, letter in np.ndenumerate(self.grid):
            for ajd_index, adj_value in common.get_adjacent(self.grid, index):
                weight = WEIGHTS[letter]
                if WEIGHTS[adj_value] <= weight + 1:
                    self.digraph.add_edge(index, ajd_index)

    def solve(self) -> int:
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
                attempt = nx.shortest_path_length(
                    self.digraph, source=point, target=self.end_coord
                )
            except nx.exception.NetworkXNoPath:
                continue
            if attempt <= distance:
                distance = attempt
        return distance


# part 1
print(Graph(parsers.lines(loader.get()), 'S').solve())  # 449
# part 2
print(Graph(parsers.lines(loader.get()), 'a').solve())  # 443
