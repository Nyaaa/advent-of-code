from collections import deque
from tools import parsers, loader, common
import numpy as np
import networkx as nx
from string import ascii_uppercase, ascii_lowercase
from typing import NamedTuple


class State(NamedTuple):
    location: tuple
    keys: dict
    steps: int
    graph: nx.Graph
    path: str


class Maze:
    def __init__(self, data: list[str]):
        mp = np.asarray([list(i) for i in data])
        self.G = nx.Graph()
        self.G.add_nodes_from([(loc, {'label': val}) for loc, val in np.ndenumerate(mp) if val != '#'])
        for loc, attr in self.G.nodes(data=True):
            for i, val in common.get_adjacent(mp, loc):
                if val == '#':
                    continue
                elif val in ascii_uppercase or attr['label'] in ascii_uppercase:
                    w = float('inf')
                else:
                    w = 1
                self.G.add_edge(loc, i, weight=w)
        self.start = next(i for i, node in self.G.nodes(data=True) if node['label'] == '@')
        self.keys = [(i, l['label']) for i, l in self.G.nodes(data=True) if l['label'] in ascii_lowercase]

    def find_path(self):
        init_keys = {k: v for k, v in self.keys}
        paths = deque([State(self.start, init_keys, 0, self.G, '')])
        solution = float('inf')
        while paths:
            loc, keys, distance, G, path = paths.popleft()
            # print(path)
            if len(path) == len(self.keys):
                if distance < solution:
                    solution = distance
                continue
            if distance > solution:
                continue
            for key, letter in keys.items():
                _distance = nx.dijkstra_path_length(G, loc, key)
                if _distance != float('inf'):
                    _G = G.copy()
                    try:
                        door = next(i for i, node in _G.nodes(data=True) if node['label'] == letter.upper())
                        _G.add_edges_from(_G.edges(door), weight=1)
                    except StopIteration:
                        pass
                    _keys = keys.copy()
                    del _keys[key]
                    paths.appendleft(State(key, _keys, distance + _distance, _G, path + letter))
        return solution

    def part_1(self):
        """
        >>> print(Maze(parsers.lines('test.txt')).part_1())
        86

        >>> print(Maze(parsers.lines('test2.txt')).part_1())
        132

        >>> print(Maze(parsers.lines('test4.txt')).part_1())
        81"""
        return self.find_path()


print(Maze(parsers.lines('test3.txt')).part_1())
# print(Maze(parsers.lines(loader.get())).part_1())
