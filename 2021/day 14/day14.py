from collections import Counter
from tools import parsers, loader


class Polymer:
    def __init__(self, data):
        self.template: str = data[0][0]
        self.rules: dict = {i: j for i, j in (x.split(' -> ') for x in data[1])}

    def insert(self, init_line: str):
        line = ''
        for i in range(len(init_line)):
            try:
                pair = init_line[i:i + 2]
                if len(pair) == 1:
                    line += pair
                    return line
                line += init_line[i] + self.rules[pair]
            except IndexError:
                return line

    def part_1(self):
        """
        >>> print(Polymer(parsers.blocks('test.txt')).part_1())
        1588"""
        step = 0
        while step != 10:
            self.template = self.insert(self.template)
            step += 1
        common = sorted(list(Counter(self.template).values()))
        return common[-1] - common[0]


print(Polymer(parsers.blocks(loader.get())).part_1())  # 2223

