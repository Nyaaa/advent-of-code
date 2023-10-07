import re

import networkx as nx

from tools import loader, parsers


class Cave:
    def __init__(self, data: list[str]) -> None:
        self.cavern = nx.Graph()

        for line in data:
            res = re.findall(r'[A-Z]+[A-Z]|\d+', line)
            node, conn = res[0], res[2:]
            self.cavern.add_node(node, flowrate=int(res[1]))
            for i in conn:
                self.cavern.add_edge(node, i, weight=1)

        self.distances = nx.floyd_warshall(self.cavern)
        self.traverse = [i for i in self.cavern.nodes if self.cavern.nodes[i]['flowrate'] > 0]

    def path_finder(
            self,
            path: list,
            time: int,
            done: list = None,
            paths: list = None
    ) -> list[list[str]]:
        if paths is None:
            paths = []
        done = path.copy() if done is None else done.copy()

        paths.append(path)
        for node in self.traverse:
            distance = self.distances[path[-1]][node]
            if distance < time and node not in done:
                self.path_finder(path + [node], time - distance - 1, done + [node], paths)
        return paths

    def total_flow(self, path: list[str], time: int) -> int:
        pressure = 0
        for i in range(1, len(path)):
            time = time - self.distances[path[i - 1]][path[i]] - 1
            pressure += time * self.cavern.nodes[path[i]]['flowrate']
        return pressure

    def part_1(self) -> int:
        """test part 1:
        >>> print(Cave(parsers.lines('test.txt')).part_1())
        1651"""
        paths = self.path_finder(['AA'], 30)
        pressure = 0
        for path in paths:
            pressure = max(pressure, self.total_flow(path, 30))
        return pressure

    def part_2(self) -> int:
        """test part 2:
        >>> print(Cave(parsers.lines('test.txt')).part_2())
        1707"""
        paths = self.path_finder(['AA'], 26)
        pressure = 0
        filtered = {}
        for path in paths:
            s = frozenset(path)
            if s not in filtered:
                filtered[s] = 0
            filtered[s] = max(filtered[s], self.total_flow(path, 26))

        for path1 in filtered:
            for path2 in filtered:
                combo = set().union(path1, path2)
                if len(combo) + 1 >= len(path1) + len(path2):
                    pressure = max(pressure, filtered[path1] + filtered[path2])

        return pressure


print(Cave(parsers.lines(loader.get())).part_1())  # 1474
print(Cave(parsers.lines(loader.get())).part_2())  # 2100
