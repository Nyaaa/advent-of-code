from tools import parsers
import networkx as nx
import re
import operator


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

        for line in data:
            res = re.findall(r"[A-Z]+[A-Z]|\d+", line)
            node, conn = res[0], res[2:]
            self.cavern.add_node(res[0], flowrate=int(res[1]), open=False)
            for i in conn:
                self.cavern.add_edge(node, i, weight=1)

    def part_1(self):
        source = 'AA'
        # flow = nx.get_node_attributes(self.cavern, 'flowrate')
        # top_flow = dict(sorted(flow.items(), key=lambda item: item[1], reverse=True))
        # print(top_flow)
        timer = 30
        value = 0
        for node in self.cavern.nodes:
            if not self.cavern.nodes[node]['open']:
                print(node)
                dist = nx.shortest_path_length(self.cavern, source=source, target=node)
                print(dist)


c = Cave(parsers.lines('test.txt')).part_1()


