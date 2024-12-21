import re
from functools import cache, partial
from itertools import pairwise

import networkx as nx
import numpy as np
from numpy.typing import NDArray

from tools import common, loader, parsers


class Numpad:
    def __init__(self, buttons: NDArray) -> None:
        self.graph = nx.Graph()
        for i, button in np.ndenumerate(buttons):
            self.graph.add_node(button, pos=complex(*i))
            for adj_i, adj_val in common.get_adjacent(buttons, i):
                self.graph.add_node(adj_val, pos=complex(*adj_i))
                self.graph.add_edge(button, adj_val)
        self.graph.remove_node('.')
        self.attrs = nx.get_node_attributes(self.graph, 'pos')
        self.directions = {-1: '^', 1: 'v', -1j: '<', 1j: '>'}

    def get_shortest_paths(self, button_a: str, button_b: str) -> list[str]:
        paths = []
        for path in nx.all_shortest_paths(self.graph, button_a, button_b):
            path_chars = [self.directions[self.attrs[b] - self.attrs[a]]
                          for a, b in pairwise(path)]
            paths.append(''.join([*path_chars, 'A']))
        return paths


numpad = Numpad(np.array([['7', '8', '9'],
                          ['4', '5', '6'],
                          ['1', '2', '3'],
                          ['.', '0', 'A']], dtype=str))
keypad = Numpad(np.array([['.', '^', 'A'], ['<', 'v', '>']], dtype=str))


@cache
def get_min_sequence_length(code: str, recursion_level: int, max_depth: int) -> int:
    result = 0
    current = 'A'
    pad = numpad if recursion_level == 0 else keypad
    for button in code:
        paths = pad.get_shortest_paths(current, button)
        lengths = partial(get_min_sequence_length,
                          recursion_level=recursion_level + 1,
                          max_depth=max_depth)
        fn = len if recursion_level == max_depth else lengths
        result += min(map(fn, paths))
        current = button
    return result


def code_complexity(data: list[str], robots: int) -> int:
    """
    >>> print(code_complexity(['029A'], 2))
    1972
    >>> print(code_complexity(parsers.lines('test.txt'), 2))
    126384"""
    result = 0
    for code in data:
        length = get_min_sequence_length(code, 0, robots)
        num = int(re.search(r'\d+', code).group(0))
        result += length * num
    return result


print(code_complexity(parsers.lines(loader.get()), robots=2))  # 162740
print(code_complexity(parsers.lines(loader.get()), robots=25))  # 203640915832208
