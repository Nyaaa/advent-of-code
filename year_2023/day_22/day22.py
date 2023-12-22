import re
from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations

from tools import loader, parsers


@dataclass
class Brick:
    coords: set[tuple[int, int, int]]

    @property
    def level(self) -> int:
        return min(i[2] for i in self.coords)

    def drop(self, grid: set) -> int:
        drop_height = 0
        while True:
            dropped = {(i[0], i[1], i[2] - 1) for i in self.coords}
            if self.level > 1 and not grid.intersection(dropped):
                self.coords = dropped
                drop_height += 1
            else:
                break
        return drop_height


def drop_all(bricks: list[Brick]) -> int:
    settled = set()
    bricks_dropped = 0
    for brick in bricks:
        if brick.drop(settled):
            bricks_dropped += 1
        settled |= brick.coords
    return bricks_dropped


def start(data: list[str]) -> tuple[int, int]:
    """
    >>> print(start(parsers.lines('test.txt')))
    (5, 7)"""
    bricks = []
    for line in data:
        x1, y1, z1, x2, y2, z2 = map(int, re.findall(r'\d+', line))
        cells = {(x, y, z)
                 for x in range(x1, x2 + 1)
                 for y in range(y1, y2 + 1)
                 for z in range(z1, z2 + 1)}
        bricks.append(Brick(cells))
    bricks = sorted(bricks, key=lambda b: b.level)
    drop_all(bricks)
    part1 = part2 = 0
    for c in combinations(bricks, len(bricks) - 1):
        bricks_dropped = drop_all(deepcopy(c))
        if not bricks_dropped:
            part1 += 1
        part2 += bricks_dropped
    return part1, part2


print(start(parsers.lines(loader.get())))  # 441, 80778
