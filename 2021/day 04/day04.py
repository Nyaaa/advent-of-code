from tools import parsers, loader
import numpy as np


class Bingo:
    def __init__(self, data):
        self.numbers = parsers.generator([i for i in data[0][0].split(',')])
        grids = [[i.split() for i in n] for n in data[1:]]
        self.grids = [np.array(i, dtype='U3') for i in grids]

    def part_1(self):
        """
        >>> print(Bingo(parsers.blocks('test.txt')).part_1())
        4512"""
        while True:
            num = next(self.numbers)
            for grid in self.grids:
                grid[grid == num] = num + '*'  # find & mark the drawn number in array
                marks = np.char.find(grid, '*')  # -1 if unmarked
                if np.any(np.all(marks >= 0, axis=1)) or np.any(np.all(marks >= 0, axis=0)):  # any full row/col marked
                    unmarked = (marks < 0).flatten()  # 1D array of bools
                    grid = grid.compress(unmarked).astype(int)  # remove marked with a mask
                    return sum(grid) * int(num)


print(Bingo(parsers.blocks(loader.get())).part_1())  # 35711
