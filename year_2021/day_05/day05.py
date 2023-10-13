import re
from itertools import zip_longest
from operator import itemgetter

import numpy as np

from tools import loader, parsers


class Vents:
    def __init__(self, data: list[str]) -> None:
        self.lines = [[(a, b), (c, d)] for a, b, c, d in
                      [list(map(int, re.findall(r'\d+', i))) for i in data]]
        flat_list = [item for sublist in self.lines for item in sublist]
        row = max(flat_list, key=itemgetter(0))[0]
        col = max(flat_list, key=itemgetter(1))[1]
        self.map = np.zeros((row + 1, col + 1), dtype=int)

    def get_points(self, part2: bool) -> list[tuple[int, int]]:
        points = []
        for start, end in self.lines:
            if start[1] == end[1]:
                row = range(min(start[0], end[0]), max(start[0], end[0]) + 1)
                points += list(zip_longest(row, [start[1]], fillvalue=start[1]))
            elif start[0] == end[0]:
                col = range(min(start[1], end[1]), max(start[1], end[1]) + 1)
                points += list(zip_longest([start[0]], col, fillvalue=start[0]))
            elif part2 and start[1] != end[1] and start[0] != end[0]:
                row_step = 1 if start[0] < end[0] else -1
                col_step = 1 if start[1] < end[1] else -1
                diagonal = [start]
                while start != end:
                    start = (start[0] + row_step, start[1] + col_step)
                    diagonal.append(start)
                points += diagonal
            else:
                continue
        return points

    def start(self, part2: bool) -> int:
        """ (x, y) = col, row
        >>> print(Vents(parsers.lines('test.txt')).start(part2=False))
        5

        >>> print(Vents(parsers.lines('test.txt')).start(part2=True))
        12"""
        for point in self.get_points(part2=part2):
            self.map[point[1]][point[0]] += 1
        return np.count_nonzero(self.map[self.map > 1])


print(Vents(parsers.lines(loader.get())).start(part2=False))  # 5632
print(Vents(parsers.lines(loader.get())).start(part2=True))  # 22213
