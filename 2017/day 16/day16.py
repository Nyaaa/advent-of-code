import numpy as np
from tools import parsers, loader
from string import ascii_lowercase


class Dance:
    def __init__(self, data: str, length: int = 16) -> None:
        self.commands = data.split(',')
        letters = [i for i in ascii_lowercase[:length]]
        self.progs = np.asarray(letters, dtype=str)
        self.initial = self.progs.copy()

    def part_1(self) -> str:
        """
        >>> print(Dance('s1,x3/4,pe/b', 5).part_1())
        baedc"""
        for c in self.commands:
            command = c[0]
            val = c[1:]
            if command == 's':
                val = int(val)
                self.progs = np.concatenate((self.progs[-val:], self.progs[:-val]))
            else:
                a, b = val.split('/')
                if command == 'x':
                    a, b = int(a), int(b)
                else:
                    a, b = np.where(self.progs == a), np.where(self.progs == b)
                self.progs[a], self.progs[b] = self.progs[b], self.progs[a]
        return ''.join(self.progs)

    def part_2(self) -> str:
        """
        >>> print(Dance('s1,x3/4,pe/b', 5).part_2())
        abcde"""
        res = ''
        times = 1_000_000_000
        i = 0
        while i != times:
            i += 1
            res = self.part_1()
            if np.array_equal(self.progs, self.initial):
                times %= i
                i = 0
        return res


print(Dance(parsers.string(loader.get())).part_1())  # pkgnhomelfdibjac
print(Dance(parsers.string(loader.get())).part_2())  # pogbjfihclkemadn
