from collections import defaultdict

import networkx as nx

from tools import loader, parsers
from year_2019 import intcode

TRACEBACK = {1: 2, 2: 1, 3: 4, 4: 3}
DIRECTIONS = {1: -1j, 2: 1j, 3: -1, 4: 1}


class RepairDroid:
    """1: north, 2: south, 3: west, 4: east"""
    def __init__(self) -> None:
        self.pc = intcode.Intcode(parsers.lines(loader.get()))
        self.maze = defaultdict(lambda: float('inf'))
        self.current = 0j
        self.target = None
        self.G = nx.Graph()

    def discover_map(self, prev_dir: int = 0, prev_pos: complex = 0j) -> None:
        for direction, vector in DIRECTIONS.items():
            pos = self.current + vector
            if self.maze[pos] == float('inf'):
                result = self.pc.run([direction])[0]  # 0: wall, 1: open
                self.maze[pos] = result
                if result == 2:
                    self.target = pos
                if result != 0j:
                    self.G.add_edge(pos, prev_pos)
                    self.current += vector
                    self.discover_map(direction, pos)
        if self.current != 0j:
            rev = TRACEBACK[prev_dir]
            self.pc.run([rev])
            self.current += DIRECTIONS[rev]

    def start(self) -> tuple[int, int]:
        self.discover_map()
        part1 = nx.shortest_path_length(self.G, 0j, self.target)
        part2 = max(
            j for i, j in nx.single_source_shortest_path_length(self.G, self.target).items()
        )
        return part1, part2


print(RepairDroid().start())  # 238, 392
