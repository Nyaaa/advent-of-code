from collections import Counter
from itertools import chain

from tools import loader, parsers

test = ['30373', '25512', '65332', '33549', '35390']
d = parsers.lines(loader.get())


class Forest:
    def __init__(self, data: list[str]) -> None:
        self.grid = [[int(i) for i in row] for row in data]
        self.x = len(self.grid[0])
        self.y = len(self.grid)
        self.tree = None

    def get_score(self, side: list[int], direction: int) -> int:
        """part 2"""

        _score = 0
        if direction == 1:
            side = side[::-1]  # iterating backwards for left & up

        for i in side:
            if i < self.tree:
                _score += 1
            else:
                _score += 1
                break

        return _score

    def is_visible(self, _x: int, _y: int) -> tuple[str, int]:
        """part 1"""

        # getting edges
        if _x in (0, self.x - 1) or _y in (0, self.y - 1):
            return 'Visible', 0

        # getting rows
        left = self.grid[_y][:_x]
        right = self.grid[_y][_x + 1:]

        # getting columns
        up = [self.grid[i][_x] for i in range(0, _y)]
        down = [self.grid[i][_x] for i in range(_y + 1, self.y)]

        s_right = self.get_score(right, 0)
        s_left = self.get_score(left, 1)
        s_up = self.get_score(up, 1)
        s_down = self.get_score(down, 0)
        _score = s_right * s_left * s_up * s_down

        if (self.tree > max(left)
                or self.tree > max(right)
                or self.tree > max(up)
                or self.tree > max(down)):
            return 'Visible', _score

        return 'Hidden', _score

    def solve(self, part: int) -> int:
        """test part 1:
        >>> print(Forest(test).solve(1))
        21

        test part 2:
        >>> print(Forest(test).solve(2))
        8"""

        vis = []
        scores = []
        for _x in range(self.x):
            vis_row = []
            score_row = []
            for _y in range(self.y):
                self.tree = self.grid[_x][_y]
                visible, score = self.is_visible(_y, _x)
                vis_row.append(visible)
                score_row.append(score)
            vis.append(vis_row)
            scores.append(score_row)

        if part == 1:
            return Counter(list(chain(*vis)))['Visible']
        else:
            return max(list(chain(*scores)))


print(Forest(d).solve(1))  # 1543
print(Forest(d).solve(2))  # 595080
