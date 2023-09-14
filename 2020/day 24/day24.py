import re
from tools.common import Point
from tools import parsers, loader


VECTORS = {'e': Point(0, 1), 'w': Point(0, -1),
           'se': Point(1, 0), 'sw': Point(1, -1),
           'nw': Point(-1, 0), 'ne': Point(-1, 1)}


class Path:
    """
    >>> print(Path('esenee'))
    ['e', 'se', 'ne', 'e']

    >>> print(Path('nwwswee').target)
    (0, 0)"""
    directions = re.compile(r'e|se|sw|w|nw|ne')

    def __init__(self, line: str) -> None:
        self.nodes = re.findall(self.directions, line)
        self.v = [VECTORS[i] for i in self.nodes]
        self.target = sum(self.v)

    def __repr__(self) -> str:
        return str(self.nodes)


class Tiles:
    def __init__(self, data: list[str]) -> None:
        self.paths = [Path(i) for i in data]
        self.tile_state = {key.target: False for key in self.paths}

    def part_1(self) -> int:
        """
        >>> print(Tiles(parsers.lines('test.txt')).part_1())
        10"""
        for i in self.paths:
            self.tile_state[i.target] = not self.tile_state[i.target]
        return sum(self.tile_state.values())

    def calculate_tile_state(self, tile: Point) -> bool:
        adjacent = [tile + i for i in VECTORS.values()]
        adj_flipped = sum(self.tile_state.get(i) or False for i in adjacent)
        tile_state = self.tile_state.get(tile) or False
        if (not tile_state and adj_flipped == 2) or \
                (tile_state and (adj_flipped == 0 or adj_flipped > 2)):
            tile_state = not tile_state
        return tile_state

    def part_2(self) -> int:
        """
        >>> print(Tiles(parsers.lines('test.txt')).part_2())
        2208"""
        self.part_1()
        _min = min(self.tile_state.keys(), key=lambda x: x.row).row - 10
        _max = max(self.tile_state.keys(), key=lambda x: x.col).col + 10
        for _ in range(100):
            floor = self.tile_state.copy()
            _min -= 1
            _max += 1
            floor_size = range(_min, _max)
            for row in floor_size:
                for col in floor_size:
                    tile = Point(row, col)
                    floor[tile] = self.calculate_tile_state(tile)
            self.tile_state = floor
        return sum(self.tile_state.values())


print(Tiles(parsers.lines(loader.get())).part_1())  # 465
print(Tiles(parsers.lines(loader.get())).part_2())  # 4078
