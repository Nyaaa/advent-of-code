import re
from tools import parsers, loader, common
import numpy as np
from operator import itemgetter

np.set_printoptions(linewidth=np.inf)


class Paper:
    def __init__(self, data):
        self.instructions = []
        points = [(int(col), int(row)) for row, col in (i.split(',') for i in data[0])]
        max_rows = max(points, key=itemgetter(0))[0] + 1
        max_cols = max(points, key=itemgetter(1))[1] + 1
        if max_rows % 2 == 0:
            max_rows += 1  # this shouldn't work, but it works
        self.sheet = np.zeros(dtype=int, shape=(max_rows, max_cols))
        for point in points:
            self.sheet[point] = 1

        pattern = re.compile(r'(\w)=(\d+)')
        for i in data[1]:
            p = pattern.search(i)
            self.instructions.append((p.group(1), int(p.group(2))))

    def fold(self, instruction: tuple[str, int]):
        axis = 1 if instruction[0] == 'x' else 0
        index = instruction[1]

        split = np.array_split(self.sheet, [index, index + 1], axis=axis)
        side_b = np.flip(split[2], axis=axis)
        self.sheet = split[0] | side_b

    def part_1(self):
        """
        >>> print(Paper(parsers.blocks('test.txt')).part_1())
        17"""
        self.fold(self.instructions[0])
        return np.count_nonzero(self.sheet)

    def part_2(self):
        for i in self.instructions:
            self.fold(i)
        return common.convert_to_image(self.sheet)


print(Paper(parsers.blocks(loader.get())).part_1())  # 775
print(Paper(parsers.blocks(loader.get())).part_2())  # REUPUPKR
