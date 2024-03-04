import numpy as np

from tools import loader, parsers

# np.set_printoptions(threshold=np.inf, linewidth=200)


class Cave:
    def __init__(self, data: list[str], part: int) -> None:
        """Right border is hardcoded as an arbitrarily large value,
        since adding dynamic stretching adds extra complexity to the code.
        May need to increase it if code fails."""
        self.left, self.right = 500, 700
        self.cavern = np.zeros((1, self.right), dtype=int)
        self.draw_cavern(
            [[list(map(int, coord.split(','))) for coord in line.split(' -> ')] for line in data]
        )
        if part == 2:
            self.increase_depth(1, 0)
            self.increase_depth(1, 1)

    def draw_cavern(self, data_clean: list[list[list[int]]]) -> None:
        for line in data_clean:
            for i, (x, y) in enumerate(line):
                try:
                    next_x, next_y = line[i + 1][0], line[i + 1][1]
                except IndexError:
                    break
                max_x, min_x, max_y = max(x, next_x), min(x, next_x), max(y, next_y)
                self.left = min(min_x, self.left)
                self.right = max(max_x, self.right)
                if max_y > len(self.cavern):
                    add = max_y + 1 - len(self.cavern)
                    self.increase_depth(add, 0)

                slice_x = np.s_[y, next_x:x + 1] if x - next_x > 0 else np.s_[y, x:next_x + 1]
                self.cavern[slice_x] = 1

                slice_y = np.s_[next_y:y + 1, x] if y - next_y > 0 else np.s_[y:next_y + 1, x]
                self.cavern[slice_y] = 1

    def increase_depth(self, lines: int, value: int) -> None:
        self.cavern = np.pad(self.cavern, [(0, lines), (0, 0)], constant_values=value)

    def fall(self, row: int, column: int) -> None:
        if self.cavern[0][column] == 1:
            raise IndexError('reached the top')
        if self.cavern[row][column] != 1:
            self.fall(row + 1, column)
        elif self.cavern[row][column - 1] != 1:
            self.fall(row, column - 1)
        elif self.cavern[row][column + 1] != 1:
            self.fall(row, column + 1)
        else:
            self.cavern[row - 1][column] = 1
            self.left = min(self.left, column)
            self.right = max(self.right, column)

    def go(self) -> int:
        """test part 1:
        >>> print(Cave(parsers.lines('test.txt'), part=1).go())
        24

        test part 2:
        >>> print(Cave(parsers.lines('test.txt'), part=2).go())
        93"""
        counter = 0
        while True:
            try:
                self.fall(0, 500)
                counter += 1
            except IndexError:
                break
        return counter


print(Cave(parsers.lines(loader.get()), part=1).go())  # 774
print(Cave(parsers.lines(loader.get()), part=2).go())  # 22499
