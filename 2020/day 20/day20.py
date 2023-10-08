import re
from collections import Counter, defaultdict
from functools import cached_property
from itertools import permutations
from math import prod, sqrt

import numpy as np
from numpy.typing import NDArray

from tools import loader, parsers

# monster is handled as a tile to get rotations
MONSTER = ['0',
           '                  # ',
           '#    ##    ##    ###',
           ' #  #  #  #  #  #   ']


class Tile:
    def __init__(self, tile_data: list[str]) -> None:
        self.id = int(re.findall(r'\d+', tile_data[0])[0])
        self.pixels = np.array([[1 if i == '#' else 0 for i in row] for row in tile_data[1:]])

    def __repr__(self) -> str:
        return str(self.id)

    @cached_property
    def all_rotations(self) -> list[tuple[NDArray, ...]]:
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
    def all_edges(self) -> list[str]:
        return [str(element) for sublist in
                (i[1] for i in self.all_rotations) for element in sublist]

    def get_edges(self, tile: NDArray = None) -> tuple[int, ...]:
        if tile is None:
            tile = self.pixels
        return tile[0], tile[-1], tile[:, 0], tile[:, -1]  # top, bottom, left, right


class Jigsaw:
    def __init__(self, data: list[list[str]]) -> None:
        self.tiles = [Tile(i) for i in data]
        self.size = int(sqrt(len(self.tiles)))

    def get_matching_borders(self) -> defaultdict:
        matches = defaultdict(set)
        for tile, other in permutations(self.tiles, 2):
            for i in tile.all_edges:
                for j in other.all_edges:
                    if i == j:
                        matches[i].update([tile, other])
                    else:
                        matches[i].add(tile)
        return matches

    def get_corners(self) -> list[Tile]:
        values = self.get_matching_borders().copy().values()
        c = Counter([i.pop() for i in values if len(i) == 1])
        return [key for key, value in c.items() if value == 4]

    def part_1(self) -> int:
        """
        >>> print(Jigsaw(parsers.blocks('test.txt')).part_1())
        20899048083289"""
        return prod([i.id for i in self.get_corners()])

    def compose_map(self) -> NDArray[Tile]:
        matches = self.get_matching_borders()
        grid = np.zeros(shape=(self.size, self.size), dtype=Tile)
        for corner in self.get_corners():
            corner_edges = corner.get_edges()
            outer_edges = [i for i in corner_edges if len(matches[str(i)]) == 1]
            for tile, edges in corner.all_rotations:
                if (np.array_equal(outer_edges[0], edges[2])
                        and np.array_equal(outer_edges[1], edges[0])):
                    corner.pixels = tile
                    grid[0, 0] = corner
        for i, val in np.ndenumerate(grid):
            if val == 0:
                if i[0] == 0:  # top row, left to right
                    left: Tile = grid[i[0], i[1] - 1]
                    right_edge = left.get_edges()[3]
                    connection = [val for key, val in matches.items() if key == str(right_edge)]
                    right_tile = (connection[0] - {left}).pop()
                    for tile, edges in right_tile.all_rotations:
                        if np.array_equal(right_edge, edges[2]):
                            right_tile.pixels = tile
                    grid[i] = right_tile
                else:  # the rest, top to bottom
                    top: Tile = grid[i[0] - 1, i[1]]
                    bottom_edge = top.get_edges()[1]
                    connection = [val for key, val in matches.items() if key == str(bottom_edge)]
                    bottom_tile = (connection[0] - {top}).pop()
                    for tile, edges in bottom_tile.all_rotations:
                        if np.array_equal(bottom_edge, edges[0]):
                            bottom_tile.pixels = tile
                    grid[i] = bottom_tile
        return grid

    def part_2(self) -> int:
        """
        >>> print(Jigsaw(parsers.blocks('test.txt')).part_2())
        273"""
        grid = self.compose_map()
        stacked_rows = []
        for row in range(self.size):
            r = []
            for col in range(self.size):
                r.append(grid[row][col].pixels[1:-1, 1:-1])
            stacked_rows.append(np.hstack(r))
        final_image = np.vstack(stacked_rows)

        monster = Tile(MONSTER)
        count = 0
        for m in [i[0] for i in monster.all_rotations]:
            for row in range(final_image.shape[0] - m.shape[0]):
                for col in range(final_image.shape[1] - m.shape[1]):
                    if np.all(final_image[row:row + m.shape[0], col:col + m.shape[1]] & m == m):
                        count += 1
            if count > 0:
                return np.count_nonzero(final_image) - (count * np.count_nonzero(monster.pixels))
        return None


print(Jigsaw(parsers.blocks(loader.get())).part_1())  # 7492183537913
print(Jigsaw(parsers.blocks(loader.get())).part_2())  # 2323
