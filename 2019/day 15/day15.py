from tools import parsers, loader, intcode
import networkx as nx
from collections import defaultdict


class RepairDroid:
    """1: north, 2: south, 3: west, 4: east"""
    traceback = {1: 2, 2: 1, 3: 4, 4: 3}
    directions = {1: -1j, 2: 1j, 3: -1, 4: 1}

    def __init__(self):
        self.pc = intcode.Intcode(parsers.lines(loader.get()))
        self.maze = defaultdict(lambda: None)
        self.current = 0j
        self.target = None
        self.G = nx.Graph()

    def discover_map(self, prev_dir: int = 0, prev_pos: complex = 0j):
        for direction, vector in self.directions.items():
            pos = self.current + vector
            if self.maze[pos] is None:
                result = self.pc.run([direction])[0]  # 0: wall, 1: open
                self.maze[pos] = result
                if result == 2:
                    self.target = pos
                if result != 0j:
                    self.G.add_edge(pos, prev_pos)
                    self.current += self.directions[direction]
                    self.discover_map(direction, pos)
        if self.current != 0j:
            rev = self.traceback[prev_dir]
            self.pc.run([rev])
            self.current += self.directions[rev]

    def start(self):
        self.discover_map()
        part1 = nx.shortest_path_length(self.G, 0j, self.target)
        part2 = max(j for i, j in nx.single_source_shortest_path_length(self.G, self.target).items())
        return part1, part2


print(RepairDroid().start())  # 238, 392
