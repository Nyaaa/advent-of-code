from tools import parsers, loader


class Intcode:
    def __init__(self, data: list):
        self.data = [int(i) for i in data[0].split(',')]

    def run(self):
        """
        >>> print(Intcode(parsers.inline_test('1,1,1,4,99,5,6,0,99')).run())
        [30, 1, 1, 4, 2, 5, 6, 0, 99]"""
        i = 0
        result = self.data.copy()
        while i <= len(result):
            match op := result[i]:  # NOSONAR
                case 1 | 2:
                    index1, index2, out = result[i + 1:i + 4]
                    i += 4
                    result[out] = result[index1] + result[index2] if op == 1 else result[index1] * result[index2]
                case 99:
                    return result

    def part_1(self):
        self.data[1] = 12
        self.data[2] = 2
        return self.run()[0]

    def part_2(self):
        for i in range(100):
            for j in range(100):
                self.data[1] = i
                self.data[2] = j
                if self.run()[0] == 19690720:
                    return 100 * i + j


print(Intcode(parsers.lines(loader.get())).part_1())  # 5290681
print(Intcode(parsers.lines(loader.get())).part_2())  # 5741
