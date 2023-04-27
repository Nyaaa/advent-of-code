from collections import defaultdict
from tools import parsers, loader


class Graph:
    def __init__(self, data):
        self.nodes = defaultdict(list)
        for line in data:
            name, conn = line.split('-')
            self.nodes[name].append(conn)
            self.nodes[conn].append(name)

    def part_1(self):
        return self.nodes


print(Graph(parsers.lines('test.txt')).part_1())
