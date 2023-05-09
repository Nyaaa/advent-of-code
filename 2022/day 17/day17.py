from itertools import cycle, chain, repeat
from numpy.typing import NDArray
from tools import parsers, loader, timer
import numpy as np

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
    def __init__(self, jets):
        self.cavern = np.ones((1, 7), dtype=bool)
        self.stones = cycle(STONES)
        self.jets = roundrobin(jets)

    def spawn_stone(self, stone: list[tuple]):
        stone_height = len(set(row for row, _ in stone))
        # only checking top 10 rows
        empty_rows = np.count_nonzero(np.all(~self.cavern[:10], axis=1))
        to_add = stone_height + 3 - empty_rows
        if to_add > 0:
            # numpy padding: [(top, bottom), (left, right)] value is thickness
            self.cavern = np.pad(self.cavern, [(to_add, 0), (0, 0)], mode='constant', constant_values=0)
        elif to_add < 0:
            self.cavern = np.delete(self.cavern, np.s_[0:abs(to_add)], 0)

    def move(self, stone: list[tuple]):
        while True:
            direction = next(self.jets)
            _stone = [(unit[0] + DIRS[direction][0], unit[1] + DIRS[direction][1]) for unit in stone]

            if direction != 'd':
                if any(unit[1] < 0 or unit[1] >= 7 or self.cavern[unit] for unit in _stone):
                    _stone = stone
            else:
                if any(self.cavern[unit] for unit in _stone):
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
                found = np.all(np.all(rolling_window(self.cavern, top.shape) == top, axis=2), axis=2).nonzero()
                rez = found[0]
                if 3 >= len(rez) > 1 and rez[-1] not in matches:
                    matches[rez[-1]] = counter
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
        self.cavern = np.delete(self.cavern, np.s_[0:empty_rows], 0)
        return len(self.cavern) - 1 + skipped_height


def rolling_window(arr: NDArray, shape: tuple) -> NDArray:
    """Rolling window for 2D numpy array"""
    s = (arr.shape[0] - shape[0] + 1,) + (arr.shape[1] - shape[1] + 1,) + shape
    strides = arr.strides + arr.strides
    return np.lib.stride_tricks.as_strided(arr, shape=s, strides=strides)


def roundrobin(jets):
    """Alternate between two iterators"""
    nexts = cycle(iter(it).__next__ for it in (cycle(jets), repeat('d')))
    yield from chain.from_iterable(inext() for inext in nexts)


with timer.context():
    print(Cave(*parsers.lines(loader.get())).start(2022))  # 3059
with timer.context():
    print(Cave(*parsers.lines(loader.get())).start(1_000_000_000_000))  # 1500874635587
