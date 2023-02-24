from tools import parsers, loader
from collections import deque

test = ['1', '2', '-3', '3', '-2', '0', '4']


class Decrypt:
    def __init__(self, data):
        ints = [int(line) for line in data]
        self.code: deque = deque(enumerate(ints))  # (index, value)

    def mix(self):
        for i in range(len(self.code)):
            while self.code[0][0] != i:
                self.code.rotate(-1)

            num = self.code.popleft()
            self.code.rotate(-num[1])
            self.code.append(num)

    def count(self):
        values = []
        deenum = [i[1] for i in self.code]
        for i in (1000, 2000, 3000):
            pos = (deenum.index(0) + i) % len(deenum)
            values.append(deenum[pos])
        return values

    def part_1(self):
        """test part 1:
        >>> print(Decrypt(test).part_1())
        3"""
        self.mix()
        values = self.count()
        return sum(values)


print(Decrypt(parsers.lines(loader.get())).part_1())  # 3473
