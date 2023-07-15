import re
from collections import defaultdict, Counter
from functools import cached_property
from math import prod

from numpy.typing import NDArray
from tools import parsers, loader
import numpy as np
from itertools import permutations


class Tile:
    def __init__(self, tile_data: list):
        self.id = int(re.findall(r'\d+', tile_data[0])[0])
        self.pixels = np.array([list(1 if i == '#' else 0 for i in row) for row in tile_data[1:]])

    def __repr__(self):
        return str(self.id)

    @cached_property
    def all_rotations(self):
        tile = self.pixels
        rotations = []
        for _ in range(2):
            for _ in range(4):
                tile = np.rot90(tile)
                edges = self.get_edges(tile)
                rotations.append((tile, edges))
            tile = np.flip(tile, 0)
            edges = self.get_edges(tile)
            rotations.append((tile, edges))
        return rotations

    @cached_property
    def all_edges(self):
        return [str(element) for sublist in (i[1] for i in self.all_rotations) for element in sublist]

    @staticmethod
    def get_edges(tile: NDArray):
        return tile[0], tile[-1], tile[:, 0], tile[:, -1]


class Jigsaw:
    def __init__(self, data: list):
        self.tiles = [Tile(i) for i in data]

    def part_1(self):
        """
        >>> print(Jigsaw(parsers.blocks('test.txt')).part_1())
        20899048083289"""
        matches = defaultdict(set)
        for tile, other in permutations(self.tiles, 2):
            for index, i in enumerate(tile.all_edges):
                for j in other.all_edges:
                    if i == j:
                        matches[i].update([tile, other])
                    else:
                        matches[i].add(tile)
        c = Counter([i.pop() for i in matches.values() if len(i) == 1])
        corners = [key.id for key, value in c.items() if value == 4]
        return prod(corners)


print(Jigsaw(parsers.blocks(loader.get())).part_1())  # 7492183537913
