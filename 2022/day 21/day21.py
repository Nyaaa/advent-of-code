from tools import parsers, loader


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

    def loop(self):
        while self.unknown:
            for i in list(self.unknown.keys()):
                m1, op, m2 = self.unknown[i]
                try:
                    self.known[i] = eval(f'{self.known[m1]} {op} {self.known[m2]}')
                    del self.unknown[i]
                except KeyError:
                    pass

    def part_1(self):
        """test part 1:
        >>> print(Monkey(parsers.lines('test.txt')).part_1())
        152"""
        self.loop()
        return int(self.known['root'])

    def part_2(self):
        pass


print(Monkey(parsers.lines(loader.get())).part_1())  # 194058098264286
print(Monkey(parsers.lines('test.txt')).part_2())
