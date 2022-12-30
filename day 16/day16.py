from tools import parsers
import networkx as nx
import re


class Cave:
    def __init__(self, data):
        """test graph:
        >>> C = Cave(parsers.lines('test.txt'))
        >>> print(C.cavern.number_of_nodes())
        10
        >>> print(C.cavern.edges('AA'))
        [('AA', 'DD'), ('AA', 'II'), ('AA', 'BB')]
        """
        self.cavern = nx.Graph()
        self.path = ['AA']

        for line in data:
            res = re.findall(r"[A-Z]+[A-Z]|\d+", line)
            node, conn = res[0], res[2:]
            self.cavern.add_node(res[0], flowrate=int(res[1]), open=False)
            for i in conn:
                self.cavern.add_edge(node, i, weight=1)

    def evaluate_nodes(self, source: str, timer: int):
        time_left = 30 - timer
        values = []
        target = (-99, 'AA', 0, 0)
        for node in self.cavern.nodes:
            flow = self.cavern.nodes[node]['flowrate']
            if not self.cavern.nodes[node]['open'] and flow > 0:
                dist = nx.shortest_path_length(self.cavern, source=source, target=node)
                value = (flow - dist) * time_left
                # recursion?

                values.append((value, node, dist, flow))
        print(values)
        while values:
            candidate = values.pop()
            if (candidate[0] > target[0]) or (candidate[0] == target[0] and candidate[2] < target[2]):
                target = candidate
        print(f'target: {target[1]}')
        return target

    def part_1(self):
        """test part 1:

        1651"""
        timer = 0
        flow_per_minute = 0
        total_flow = 0
        # print(nx.floyd_warshall(self.cavern))
        while timer < 30:
            timer += 1
            print(f"{'=' * 10} {timer}")
            total_flow += flow_per_minute
            current = self.path[-1]
            _, target, dist, flow = self.evaluate_nodes(current, timer)
            if current != target:
                path = nx.shortest_path(self.cavern, source=current, target=target)
                self.path.append(path[1])
                print(f'{current} -> {target}')
            else:
                flow_per_minute += flow
                self.cavern.nodes[target]['open'] = True
                print(f'opening {target}')
            print(f'fpm: {flow_per_minute}')
        return total_flow


print(Cave(parsers.lines('test.txt')).part_1())
# print(Cave(parsers.lines('input.txt')).part_1())  # 1310 too low
