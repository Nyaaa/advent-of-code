from tools import parsers, loader
import numpy as np

np.set_printoptions(threshold=np.inf)


class Bingo:
    def __init__(self, data):
        self.numbers = [i for i in data[0][0].split(',')]
        grids = [[i.split() for i in n] for n in data[1:]]
        self.grids = [np.array(i, dtype='U3') for i in grids]
        self.wins = self.get_wins()

    def get_wins(self):
        wins = []
        won = []
        for num in self.numbers:
            for i in range(len(self.grids)):
                if i in won:
                    continue
                grid = self.grids[i]
                grid[grid == num] = num + '*'  # find & mark the drawn number in array
                marks = np.char.find(grid, '*')  # -1 if unmarked
                if np.any(np.all(marks >= 0, axis=1)) or np.any(np.all(marks >= 0, axis=0)):  # any full row/col marked
                    unmarked = (marks < 0).flatten()  # 1D array of bools
                    compressed = grid.compress(unmarked).astype(int)  # remove marked with a mask
                    result = sum(compressed) * int(num)
                    wins.append(result)
                    won.append(i)
        return wins

    def part_1(self):
        """
        >>> print(Bingo(parsers.blocks('test.txt')).part_1())
        4512"""
        return self.wins[0]

    def part_2(self):
        """
        >>> print(Bingo(parsers.blocks('test.txt')).part_2())
        1924"""
        return self.wins[-1]


b = Bingo(parsers.blocks(loader.get()))
print(b.part_1())  # 35711
print(b.part_2())  # 5586
