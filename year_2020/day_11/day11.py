import numpy as np

from tools import common, loader, parsers


class Lobby:
    def __init__(self, data: list[str]) -> None:
        # 0 = floor, 1 = free seat, 2 = occupied seat
        self.map = np.array([[0 if i == '.' else 1 for i in row] for row in data])

    def part_1(self) -> int:
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
            self.map = new_map
        return np.count_nonzero(self.map == 2)

    def line_of_sight(self, index: tuple) -> int:
        directions = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
        occupied = 0
        grid_size = self.map.shape[0] - 1
        for direction in directions:
            next_cell = (index[0] + direction[0], index[1] + direction[1])
            while (0 <= next_cell[0] <= grid_size) and (0 <= next_cell[1] <= grid_size):
                if self.map[next_cell] == 1:
                    break
                if self.map[next_cell] == 2:
                    occupied += 1
                    break
                next_cell = (next_cell[0] + direction[0], next_cell[1] + direction[1])
        return occupied

    def part_2(self) -> int:
        """
        >>> print(Lobby(parsers.lines('test.txt')).part_2())
        26"""
        while True:
            new_map = self.map.copy()
            for index, value in np.ndenumerate(self.map):
                visible_occupied = self.line_of_sight(index)
                if value == 1 and visible_occupied == 0:
                    new_map[index] = 2
                elif value == 2 and visible_occupied >= 5:
                    new_map[index] = 1
            if np.array_equal(self.map, new_map):
                break
            self.map = new_map
        return np.count_nonzero(self.map == 2)


print(Lobby(parsers.lines(loader.get())).part_1())  # 2470
print(Lobby(parsers.lines(loader.get())).part_2())  # 2259
