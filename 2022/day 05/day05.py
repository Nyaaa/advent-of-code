# [B]                     [N]     [H]
# [V]         [P] [T]     [V]     [P]
# [W]     [C] [T] [S]     [H]     [N]
# [T]     [J] [Z] [M] [N] [F]     [L]
# [Q]     [W] [N] [J] [T] [Q] [R] [B]
# [N] [B] [Q] [R] [V] [F] [D] [F] [M]
# [H] [W] [S] [J] [P] [W] [L] [P] [S]
# [D] [D] [T] [F] [G] [B] [B] [H] [Z]
#  1   2   3   4   5   6   7   8   9

import queue
from tools import parsers, loader
init_load = ['DHNQTWVB', 'DWB', 'TSQWJC', 'FJRNZTP', 'GPVJMST', 'BWFTN', 'BLDQFHVN', 'HPFR', 'ZSMBLNPH']
test_load = ['ZN', 'MCD', 'P']
test = ['move 1 from 2 to 1', 'move 3 from 1 to 3', 'move 2 from 2 to 1', 'move 1 from 1 to 2']


class Crane:
    def __init__(self, data, load):
        self.data = data
        self.stack_list = []
        self.size = len(load)
        for i in range(self.size):
            self.stack_list.append(queue.LifoQueue())
            for crate in load[i]:
                self.stack_list[i].put(crate)

    def get_moves(self):
        for move in self.data:
            move = move.split(' ')
            yield int(move[1]), int(move[3]), int(move[5])

    def part_1(self):
        """test part 1:
        >>> print(Crane(test, test_load).part_1())
        CMZ"""
        for amount, _from, _to in self.get_moves():
            for _ in range(amount):
                crate = self.stack_list[_from - 1].get()
                self.stack_list[_to - 1].put(crate)
        return self.result()

    def part_2(self):
        """test part 1:
        >>> print(Crane(test, test_load).part_2())
        MCD"""
        for amount, _from, _to in self.get_moves():
            crates = []
            for _ in range(amount):
                crates.insert(0, self.stack_list[_from - 1].get())
            for crate in crates:
                self.stack_list[_to - 1].put(crate)
        return self.result()

    def result(self):
        message = ""
        for stack in self.stack_list:
            message += stack.get()
        return message


print(Crane(parsers.lines(loader.get()), init_load).part_1())  # PSNRGBTFT
print(Crane(parsers.lines(loader.get()), init_load).part_2())  # BNTZFPMMW
