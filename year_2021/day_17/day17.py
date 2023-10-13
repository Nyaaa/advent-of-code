import re

from tools import loader, parsers


class Probe:
    def __init__(self, data: str) -> None:
        target_area = re.findall(r'\d+|-\d+', data)
        # x -> cols, y -> rows
        self.col_min, self.col_max, self.row_min, self.row_max = (int(i) for i in target_area)

    def simulate(self, row_speed: int, col_speed: int) -> int | None:
        row, col = 0, 0
        max_height = 0

        while col < self.col_max and row > self.row_min:
            col += col_speed
            row += row_speed

            row_speed -= 1
            if col_speed > 0:
                col_speed -= 1
            elif col_speed < 0:
                col_speed += 1

            max_height = row if row > max_height else max_height
            if self.row_min <= row <= self.row_max and self.col_min <= col <= self.col_max:
                return max_height
        return None

    def start(self) -> tuple[int, int]:
        """
        >>> print(Probe('target area: x=20..30, y=-10..-5').start())
        (45, 112)"""
        heights = []
        options = 0
        for col_speed in range(self.col_max + 1):
            # upper limit may or may not be enough
            for row_speed in range(self.row_min, abs(self.row_max) + 100):
                height = self.simulate(row_speed, col_speed)
                if height is not None:
                    heights.append(height)
                    options += 1
        return max(heights), options


print(Probe(parsers.string(loader.get())).start())  # part 1: 9870, part 2: 5523
