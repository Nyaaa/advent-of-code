import re
from tools import parsers, loader
import numpy as np
from operator import itemgetter


class Paper:
    def __init__(self, data):
        self.instructions = []
        points = [(int(col), int(row)) for row, col in (i.split(',') for i in data[0])]
        max_rows = max(points, key=itemgetter(0))[0] + 1
        max_cols = max(points, key=itemgetter(1))[1] + 1
        self.sheet = np.zeros(dtype=int, shape=(max_rows, max_cols))
        for point in points:
            self.sheet[point] = 1

        pattern = re.compile(r'(\w)=(\d)')
        for i in data[1]:
            p = pattern.search(i)
            self.instructions.append((p.group(1), int(p.group(2))))

    def part_1(self):
        return self.instructions


print(Paper(parsers.blocks('test.txt')).part_1())
