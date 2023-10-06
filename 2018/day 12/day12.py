import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from tools import loader, parsers


class Plants:
    def __init__(self, data: list[str]) -> None:
        data = [i.replace('#', '1').replace('.', '0') for i in data]
        self.current_state = np.array([int(i) for i in data[0].split(': ')[1]], dtype=int)
        self.zero = 0
        self.rules = {}
        for rule in data[2:]:
            condition, result = rule.split(' => ')
            arr = np.array([int(i) for i in condition], dtype=int)
            self.rules[arr.tobytes()] = int(result)

    def grow(self) -> None:
        if np.sum(self.current_state[-5:]) > 0:
            self.current_state = np.pad(self.current_state, 5)
            self.zero += 5
        new_state = self.current_state.copy()
        for i, window in enumerate(sliding_window_view(self.current_state, (5,))):
            new_state[i + 2] = self.rules.get(window.tobytes(), 0)
        self.current_state = new_state

    def part_1(self) -> int:
        """
        >>> print(Plants(parsers.lines('test.txt')).part_1())
        325"""
        for _ in range(20):
            self.grow()
        return sum(i - self.zero for i in np.where(self.current_state == 1)[0])

    def part_2(self) -> int:
        """
        >>> print(Plants(parsers.lines('test.txt')).part_2())
        999999999374"""
        _sum = 0
        diff = 0
        for i in range(200):
            self.grow()
            new_sum = sum(i - self.zero for i in np.where(self.current_state == 1)[0])
            diff = new_sum - _sum
            _sum = new_sum
        return (50000000000 - 200) * diff + _sum


print(Plants(parsers.lines(loader.get())).part_1())  # 3472
print(Plants(parsers.lines(loader.get())).part_2())  # 2600000000919
