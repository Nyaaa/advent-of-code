import re

from tools import parsers, loader


class Path:
    """
    >>> print(Path('esenee'))
    ['e', 'se', 'ne', 'e']

    >>> print(Path('nwwswee').target)
    (0, 0)"""
    directions = re.compile(r'e|se|sw|w|nw|ne')
    vectors = {'e': (0, 1), 'w': (0, -1),
               'se': (1, 0), 'sw': (1, -1),
               'nw': (-1, 0), 'ne': (-1, 1)}

    def __init__(self, line: str):
        self.nodes = re.findall(self.directions, line)
        self.v = [self.vectors[i] for i in self.nodes]
        self.target = self.get_location()

    def get_location(self) -> tuple[int, int]:
        left = self.v[0]
        for i in self.v[1:]:
            left = left[0] + i[0], left[1] + i[1]
        return left

    def __repr__(self):
        return str(self.nodes)


class Tiles:
    def __init__(self, data: list):
        self.paths = [Path(i) for i in data]
        self.tile_state = {key.target: False for key in self.paths}

    def part_1(self) -> int:
        """
        >>> print(Tiles(parsers.lines('test.txt')).part_1())
        10"""
        for i in self.paths:
            self.tile_state[i.target] = not self.tile_state[i.target]
        return sum(self.tile_state.values())


print(Tiles(parsers.lines(loader.get())).part_1())  # 465
