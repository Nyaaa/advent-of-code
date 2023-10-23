import re
from collections import deque
from itertools import zip_longest

from more_itertools import minmax

from tools import loader, parsers


class Scan:
    def __init__(self, data: list[str]) -> None:
        self.sources = deque([500+0j])
        self.clay = set()
        self.water = set()
        self.overflow = set()
        for line in data:
            _x, _y = sorted(re.findall(r'([xy])=(\d+)(?:..)?(\d+)?', line))
            if _x[2]:
                fill = int(_y[1])
                y = [fill]
                x = range(int(_x[1]), int(_x[2]) + 1)
            else:
                fill = int(_x[1])
                x = [fill]
                y = range(int(_y[1]), int(_y[2]) + 1)
            self.clay = self.clay.union(
                {complex(i, j) for i, j in zip_longest(x, y, fillvalue=fill)})
        self.top, self.bottom = (i.imag for i in minmax(self.clay, key=lambda i: i.imag))

    def flow_sideways(
            self, water: complex, direction: int, line: set = None
    ) -> tuple[bool, set[complex]]:

        if line is None:
            line = set()
        water += direction
        if water not in self.clay:
            below = water + 1j
            if below not in self.water and below not in self.clay:  # edge of basin
                if water not in self.sources:
                    self.sources.append(water)
                return False, line
            line.add(water)
            return self.flow_sideways(water, direction, line)
        return True, line

    def find_bottom(self, water: complex) -> complex:
        self.overflow.add(water)
        down = water + 1j
        if down.imag > self.bottom:
            raise ValueError('Reached bottom')
        if down not in self.clay:
            return self.find_bottom(down)
        return water

    def fill_basin(self, water: complex) -> None:
        filled_left, line_left = self.flow_sideways(water, -1)
        filled_right, line_right = self.flow_sideways(water, 1)

        if filled_left and filled_right:
            self.water.add(water)
            self.water.update(line_left | line_right)
            self.fill_basin(water - 1j)
        else:
            self.overflow.add(water)
            self.overflow.update(line_left | line_right)

    def simulate(self) -> tuple[int, int]:
        """
        >>> print(Scan(parsers.lines('test.txt')).simulate())
        (57, 29)"""
        while self.sources:
            try:
                bottom = self.find_bottom(self.sources.popleft())
            except ValueError:
                continue
            self.fill_basin(bottom)

        part1 = sum(self.top <= i.imag <= self.bottom for i in self.water | self.overflow)
        part2 = sum(self.top <= i.imag <= self.bottom for i in self.water)
        return part1, part2


print(Scan(parsers.lines(loader.get())).simulate())  # 31883, 24927
