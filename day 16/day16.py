from tools import parsers
import networkx as nx
import re


class Cave:
    def __init__(self, data):
        self.cavern = nx.Graph()
        self.path = ['AA']

        for line in data:
            res = re.findall(r"[A-Z]+[A-Z]|\d+", line)
            node, conn = res[0], res[2:]
            self.cavern.add_node(res[0], flowrate=int(res[1]), open=False)
            for i in conn:
                self.cavern.add_edge(node, i, weight=1)
        self.distances = nx.floyd_warshall(self.cavern)

    def evaluate_nodes(self, traverse, path, timer: int):
        best_result = 0
        source = path[-1]
        if traverse:
            for node in traverse:
                distance = self.distances[source][node] + 1
                if distance < timer:
                    _timer = timer - distance
                    node_value = self.cavern.nodes[node]['flowrate'] * _timer
                    to_traverse = traverse.copy()
                    to_traverse.remove(node)
                    _path = path + [node]
                    best_result = max(best_result, node_value + self.evaluate_nodes(to_traverse, _path, _timer))
        return best_result

    def part_1(self):
        """test part 1:
        >>> print(Cave(parsers.lines('test.txt')).part_1())
        1651"""
        path = ['AA']
        timer = 30
        traverse = [i for i in self.cavern.nodes if
                    not self.cavern.nodes[i]['open'] and
                    self.cavern.nodes[i]['flowrate'] > 0]
        return self.evaluate_nodes(traverse, path, timer)


print(Cave(parsers.lines('input.txt')).part_1())  # 1474
