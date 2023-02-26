from tools import parsers, loader
import sys


class Monkey:
    def __init__(self, data):
        self.known = {}
        self.unknown = {}

        for line in data:
            line = line.replace(':', '').split()
            if len(line) == 2:
                self.known[line[0]] = int(line[1])
            else:
                self.unknown[line[0]] = [*line[1:]]

    def part_1(self):
        """test part 1:
        >>> print(Monkey(parsers.lines('test.txt')).part_1())
        152"""
        iterate = self.unknown.copy()
        known = self.known.copy()
        while iterate:
            for i in tuple(iterate.keys()):
                m1, op, m2 = iterate[i]
                try:
                    known[i] = eval(f'{known[m1]} {op} {known[m2]}')
                    del iterate[i]
                except KeyError:
                    pass
        return int(known['root'])

    def part_2(self):
        """Binary search
        test part 2:
        >>> print(Monkey(parsers.lines('test.txt')).part_2())
        301"""
        self.unknown['root'][1] = '-'
        left = 0
        right = sys.maxsize
        modifier = self.part_1() > 0

        while True:
            middle = (left + right) // 2
            self.known['humn'] = middle
            diff = self.part_1()

            if (diff > 0 and modifier) or (diff < 0 and not modifier):
                left = middle + 1
            elif (diff > 0 and not modifier) or (diff < 0 and modifier):
                right = middle - 1
            else:
                return middle


print(Monkey(parsers.lines(loader.get())).part_1())  # 194058098264286
print(Monkey(parsers.lines(loader.get())).part_2())  # 3592056845086
