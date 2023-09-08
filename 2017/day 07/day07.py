from collections import Counter
from tools import parsers, loader
import networkx as nx
import re


class Tower:
    def __init__(self, data: list[str]):
        self.G = nx.DiGraph()
        for line in data:
            weight = int(re.findall(r'\d+', line)[0])
            names = re.findall(r'\b[^\d\W]+\b', line)
            self.G.add_node(names[0], weight=weight, self_weight=weight)
            if len(names) > 1:
                for i in names[1:]:
                    self.G.add_edge(names[0], i)

    def part_1(self):
        """
        >>> print(Tower(parsers.lines('test.txt')).part_1())
        tknk"""
        return next(nx.lexicographical_topological_sort(self.G))

    def part_2(self):
        """
        >>> print(Tower(parsers.lines('test.txt')).part_2())
        60"""
        seen = set()
        while True:
            for node, weight in self.G.nodes(data=True):
                children = set(self.G.neighbors(node))
                if not children:
                    seen.add(node)
                    continue
                if node in seen or not all(i in seen for i in children):
                    continue
                weights = [self.G.nodes[i] for i in children]
                total_weight = [i['weight'] for i in weights]
                count = Counter(total_weight)
                if len(count) != 1:
                    self_weight = count.most_common()[-1][0]
                    self_weight = next(i['self_weight'] for i in weights if i['weight'] == self_weight)
                    return self_weight - (max(total_weight) - min(total_weight))
                if weights:
                    self.G.nodes[node]['weight'] += sum(total_weight)
                    seen.add(node)


print(Tower(parsers.lines(loader.get())).part_1())  # uownj
print(Tower(parsers.lines(loader.get())).part_2())  # 596
