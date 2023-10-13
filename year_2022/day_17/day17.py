from itertools import cycle, repeat

import numpy as np
from more_itertools import roundrobin
from numpy.lib.stride_tricks import sliding_window_view

from tools import loader, parsers, timer

np.set_printoptions(threshold=np.inf, linewidth=100)
test = '>>><<><>><<<>><>>><<<>>><<<><<<>><>><<>>'
STONES = [
    [(0, 2), (0, 3), (0, 4), (0, 5)],
    [(1, 2), (1, 3), (1, 4), (0, 3), (2, 3)],
    [(2, 2), (2, 3), (2, 4), (1, 4), (0, 4)],
    [(0, 2), (1, 2), (2, 2), (3, 2)],
    [(0, 2), (0, 3), (1, 2), (1, 3)]]
DIRS = {
    'd': (1, 0),
    '>': (0, 1),
    '<': (0, -1)}


class Cave:
    def __init__(self, jets: str) -> None:
        self.cavern = np.ones((1, 7), dtype=bool)
        self.stones = cycle(STONES)
        self.jets = roundrobin(cycle(jets), repeat('d'))

    def spawn_stone(self, stone: list[tuple[int, int]]) -> None:
        stone_height = len({row for row, _ in stone})
        # only checking top 10 rows
        empty_rows = np.count_nonzero(np.all(~self.cavern[:10], axis=1))
        to_add = stone_height + 3 - empty_rows
        if to_add > 0:
            self.cavern = np.r_[np.zeros((to_add, 7), dtype=bool), self.cavern]
        elif to_add < 0:
            self.cavern = np.delete(self.cavern, slice(0, abs(to_add)), 0)

    def move(self, stone: list[tuple[int, int]]) -> None:
        while True:
            direction = next(self.jets)
            _stone = [
                (unit[0] + DIRS[direction][0], unit[1] + DIRS[direction][1]) for unit in stone
            ]

            if direction != 'd':
                if any(unit[1] < 0 or unit[1] >= 7 or self.cavern[unit] for unit in _stone):
                    _stone = stone
            elif any(self.cavern[unit] for unit in _stone):
                for i in stone:
                    self.cavern[i] = 1
                break
            stone = _stone

    def start(self, rocks: int) -> int:
        """test part 1:
        >>> print(Cave(test).start(2022))
        3068

        test part 2:
        >>> print(Cave(test).start(1_000_000_000_000))
        1514285714288"""
        matches = {}
        counter = 0
        seq_length = 0
        skipped_height = 0

        while rocks > counter:
            stone = next(self.stones)
            self.spawn_stone(stone)
            self.move(stone)

            if not seq_length and rocks > 2022:
                top = self.cavern[5:25]  # increase the window size if fails
                found = np.all(np.all(
                    sliding_window_view(
                        self.cavern, top.shape
                    ) == top, axis=2), axis=2).nonzero()[0]
                if 3 >= len(found) > 1 and found[-1] not in matches:
                    matches[found[-1]] = counter
                if len(matches) == 2:
                    seq_length = int(np.diff(list(matches.keys())))
                    seq_stones = int(np.diff(list(matches.values())))
                    seqs = (rocks - counter) // seq_stones
                    skipped_stones = seqs * seq_stones  # test: 35, input: 1715
                    skipped_height = seqs * seq_length  # test: 53, input: 2574
                    counter += skipped_stones
            counter += 1

        # Trim empty rows at the top
        empty_rows = np.count_nonzero(np.all(~self.cavern[:20], axis=1))
        self.cavern = np.delete(self.cavern, slice(0, empty_rows), 0)
        return len(self.cavern) - 1 + skipped_height


with timer.context():
    print(Cave(*parsers.lines(loader.get())).start(2022))  # 3059
with timer.context():
    print(Cave(*parsers.lines(loader.get())).start(1_000_000_000_000))  # 1500874635587
