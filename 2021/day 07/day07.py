from tools import parsers, loader

TEST = '16,1,2,0,4,2,7,1,2,14'


class Crabs:
    def __init__(self, data):
        self.data = [int(i) for i in data[0].split(',')]

    @staticmethod
    def fuel(a: int, b: int, part2: bool) -> int:
        p1 = abs(a - b)
        return p1 if not part2 else (p1 * (p1 + 1)) // 2

    def start(self, part2: bool) -> int:
        """
        >>> print(Crabs(parsers.inline_test(TEST)).start(False))
        37

        >>> print(Crabs(parsers.inline_test(TEST)).start(True))
        168"""
        costs = [sum(self.fuel(i, j, part2) for j in self.data) for i in range(len(self.data))]
        return min(costs)


print(Crabs(parsers.lines(loader.get())).start(False))  # 359648
print(Crabs(parsers.lines(loader.get())).start(True))  # 100727924
