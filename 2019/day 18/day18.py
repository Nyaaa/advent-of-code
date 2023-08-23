from tools import parsers, loader, common
import numpy as np
import networkx as nx
from string import ascii_uppercase, ascii_lowercase


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

    def find_path(self, G, current, t, keys):
        steps = 0
        while True:
            if not t:
                options = []
                for target in keys.keys():
                    n = nx.dijkstra_path_length(G, current, target)
                    if n < float('inf'):
                        options.append(target)
                if len(options) > 1:
                    r = {}
                    for i in options:
                        results = self.find_path(G.copy(), current, i, keys.copy())
                        r[i] = results
                    t = min(r, key=r.get)
                else:
                    t = options[0]
            distance = nx.dijkstra_path_length(G, current, t)
            steps += distance
            current = t
            keys.pop(current)
            t = None
            unlocked = G.nodes[current]['label'].upper()
            try:
                door = next(i for i, node in G.nodes(data=True) if node['label'] == unlocked)
                G.add_edges_from(G.edges(door), weight=1)
            except StopIteration:
                if keys:
                    continue
                else:
                    return steps

    def part_1(self):
        """
        >>> print(Maze(parsers.lines('test.txt')).part_1())
        86

        >>> print(Maze(parsers.lines('test2.txt')).part_1())
        132"""
        current = self.start
        keys = {k: v for k, v in self.keys}
        steps = self.find_path(self.G, current, None, keys)
        return steps


print(Maze(parsers.lines('test4.txt')).part_1())
