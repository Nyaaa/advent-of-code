from tools import parsers, loader
import numpy as np
import re
from operator import itemgetter
from itertools import zip_longest

np.set_printoptions(threshold=np.inf, linewidth=np.inf)


class Vents:
    def __init__(self, data):
        self.lines = [[(a, b), (c, d)] for a, b, c, d in [list(map(int, re.findall(r'\d+', i))) for i in data]]
        flat_list = [item for sublist in self.lines for item in sublist]
        row = max(flat_list, key=itemgetter(0))[0]
        col = max(flat_list, key=itemgetter(1))[1]
        self.map = np.zeros((row + 1, col + 1), dtype=int)

    def get_points(self, part2: bool):
        points = []
        for start, end in self.lines:
            if start[1] == end[1]:
                row = range(min(start[0], end[0]), max(start[0], end[0]) + 1)
                points += list(zip_longest(row, [start[1]], fillvalue=start[1]))
            elif start[0] == end[0]:
                col = range(min(start[1], end[1]), max(start[1], end[1]) + 1)
                points += list(zip_longest([start[0]], col, fillvalue=start[0]))
            elif part2 and start[1] != end[1] and start[0] != end[0]:
                print(start, end)

            else:
                continue
        return points

    def part_1(self):
        """ (x, y) = col, row
        >>> print(Vents(parsers.lines('test.txt')).part_1())
        5"""
        for point in self.get_points(part2=False):
            self.map[point[1]][point[0]] += 1
        return np.count_nonzero(self.map[self.map > 1])

    def part_2(self):
        points = self.get_points(part2=True)
        return


print(Vents(parsers.lines(loader.get())).part_1())  # 5632
print(Vents(parsers.lines('test.txt')).part_2())
