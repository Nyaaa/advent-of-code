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

init_load = ['DHNQTWVB',
             'DWB',
             'TSQWJC',
             'FJRNZTP',
             'GPVJMST',
             'BWFTN',
             'BLDQFHVN',
             'HPFR',
             'ZSMBLNPH']


class Crane:
    def __init__(self, data):
        self.data = data
        self.stack_list = []
        for i in range(9):
            self.stack_list.append(queue.LifoQueue())

        for i in range(9):
            for crate in init_load[i]:
                self.stack_list[i].put(crate)

    def get_moves(self):
        for move in self.data:
            move = move.split(' ')
            yield int(move[1]), int(move[3]), int(move[5])

    def part_1(self):
        for i in self.get_moves():
            amount, _from, _to = i
            for _ in range(amount):
                crate = self.stack_list[_from - 1].get()
                self.stack_list[_to - 1].put(crate)
        return self.result()

    def part_2(self):
        for i in self.get_moves():
            amount, _from, _to = i
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


with open('input05.txt') as f:
    data = f.read().splitlines()

print(Crane(data).part_1())  # PSNRGBTFT
print(Crane(data).part_2())  # BNTZFPMMW
