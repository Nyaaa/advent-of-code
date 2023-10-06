import heapq
from functools import cached_property

import numpy as np

from tools import common, loader, parsers


class Maze:
    def __init__(self, data: list[str]) -> None:
        self.grid = np.asarray([list(i) for i in data], dtype=str)
        self.keys = {i: val for i, val in np.ndenumerate(self.grid) if val.islower()}
        self.locations = self.keys.copy()

    @cached_property
    def all_paths(self) -> dict[str, dict[str, tuple[int, frozenset]]]:
        paths = {}
        for pos, item in self.locations.items():
            queue = [(0, pos, frozenset())]
            seen = {pos}
            _paths = {}
            while queue:
                steps, pos, doors = heapq.heappop(queue)
                letter = self.grid[pos]
                if letter != item and letter.islower():
                    _paths[letter] = (steps, doors)
                    continue
                elif letter.isupper():
                    doors = doors.union(letter.lower())
                for next_pos, val in common.get_adjacent(self.grid, pos):
                    if next_pos not in seen and val != '#':
                        heapq.heappush(queue, (steps + 1, next_pos, doors))
                        seen.add(next_pos)
            paths[item] = _paths
        return paths

    def part_1(self) -> int:
        """
        >>> print(Maze(parsers.lines('test.txt')).part_1())
        86

        >>> print(Maze(parsers.lines('test2.txt')).part_1())
        132

        >>> print(Maze(parsers.lines('test3.txt')).part_1())
        136

        >>> print(Maze(parsers.lines('test4.txt')).part_1())
        81"""
        robots = {tuple(j): str(i) for i, j in enumerate(np.argwhere(self.grid == '@'))}
        self.locations.update(robots)
        queue = [(0, ''.join(robots.values()), frozenset())]
        seen = set()
        while queue:
            steps, current, keys = heapq.heappop(queue)
            if (current, keys) in seen:
                continue
            seen.add((current, keys))
            if len(keys) == len(self.keys):
                return steps
            for i, pos in enumerate(current):
                for next_pos, (_steps, doors) in self.all_paths[pos].items():
                    if doors - keys:
                        continue
                    _current = f'{current[:i]}{next_pos}{current[i + 1:]}'
                    heapq.heappush(queue, (steps + _steps, _current, keys | {next_pos}))

    def part_2(self) -> int:
        """
        >>> print(Maze(parsers.lines('test5.txt')).part_2())
        32"""
        row, col = np.argwhere(self.grid == '@')[0]
        more_bots = np.array([['@', '#', '@'],
                              ['#', '#', '#'],
                              ['@', '#', '@']], dtype=str)
        self.grid[row - 1:row + 2, col - 1:col + 2] = more_bots
        return self.part_1()


print(Maze(parsers.lines(loader.get())).part_1())  # 3764
print(Maze(parsers.lines(loader.get())).part_2())  # 1738
