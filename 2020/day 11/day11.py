from tools import parsers, loader, common
import numpy as np


class Lobby:
    def __init__(self, data):
        # 0 = floor, 1 = free seat, 2 = occupied seat
        self.map = np.array([list(0 if i == '.' else 1 for i in row) for row in data])

    def part_1(self):
        """
        >>> print(Lobby(parsers.lines('test.txt')).part_1())
        37"""
        while True:
            new_map = self.map.copy()
            for index, value in np.ndenumerate(self.map):
                adj = [j for _, j in common.get_adjacent(self.map, index, with_corners=True)]
                if value == 1 and all(i != 2 for i in adj):
                    new_map[index] = 2
                elif value == 2 and len([i for i in adj if i == 2]) >= 4:
                    new_map[index] = 1
            if np.array_equal(self.map, new_map):
                break
            else:
                self.map = new_map
        return np.count_nonzero(self.map == 2)


print(Lobby(parsers.lines(loader.get())).part_1())  # 2470
