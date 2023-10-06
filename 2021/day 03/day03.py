from collections import Counter

from tools import loader, parsers


class Diagnostics:
    def __init__(self, data: list[str]) -> None:
        self.data = data

    def counter(self, search: int) -> str:
        val = ''
        for i in range(len(self.data[0])):
            val += Counter([n[i] for n in self.data]).most_common()[search][0]
        return val

    def filtr(self, high: str, low: str) -> str:
        lst = self.data.copy()
        while True:
            for i in range(len(lst[0])):
                c = Counter([n[i] for n in lst])
                common = high if c['1'] > c['0'] or c['1'] == c['0'] else low
                lst = [j for j in lst if j[i] == common]
                if len(lst) == 1:
                    return lst[0]

    def part_1(self) -> int:
        """
        >>> print(Diagnostics(parsers.lines('test.txt')).part_1())
        198"""
        gamma = int(self.counter(0), 2)
        epsilon = int(self.counter(-1), 2)
        return gamma * epsilon

    def part_2(self) -> int:
        """
        >>> print(Diagnostics(parsers.lines('test.txt')).part_2())
        230"""
        oxygen = int(self.filtr('1', '0'), 2)
        co2 = int(self.filtr('0', '1'), 2)
        return oxygen * co2


print(Diagnostics(parsers.lines(loader.get())).part_1())  # 3969000
print(Diagnostics(parsers.lines(loader.get())).part_2())  # 4267809
