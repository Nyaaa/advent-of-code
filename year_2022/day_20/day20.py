from collections import deque

from tools import loader, parsers

test = ['1', '2', '-3', '3', '-2', '0', '4']


class Decrypt:
    def __init__(self, data: list[str]) -> None:
        self.ints = [int(line) for line in data]
        self.code: deque = deque(enumerate(self.ints))  # (index, value)

    def mix(self) -> None:
        for i in range(len(self.code)):
            while self.code[0][0] != i:
                self.code.rotate(-1)

            num = self.code.popleft()
            self.code.rotate(-num[1])
            self.code.append(num)

    def count(self) -> int:
        values = []
        deenum = [i[1] for i in self.code]
        for i in 1000, 2000, 3000:
            pos = (deenum.index(0) + i) % len(deenum)
            values.append(deenum[pos])
        return sum(values)

    def part_1(self) -> int:
        """test part 1:
        >>> print(Decrypt(test).part_1())
        3"""
        self.mix()
        return self.count()

    def part_2(self) -> int:
        """test part 1:
        >>> print(Decrypt(test).part_2())
        1623178306"""
        key = 811589153
        self.ints = [i * key for i in self.ints]
        self.code = deque(enumerate(self.ints))
        for _ in range(10):
            self.mix()
        return self.count()


print(Decrypt(parsers.lines(loader.get())).part_1())  # 3473
print(Decrypt(parsers.lines(loader.get())).part_2())  # 7496649006261
