from tools import loader, parsers

data = parsers.blocks(loader.get())
test = parsers.blocks('test.txt')


class Puzzle:
    def __init__(self, _data):
        self.sums = [sum(int(i) for i in block) for block in _data]

    def part_1(self):
        """test part 1:
        >>> print(Puzzle(test).part_1())
        24000"""
        return max(self.sums)

    def part_2(self):
        """test part 2:
        >>> print(Puzzle(test).part_2())
        45000"""
        self.sums.sort(reverse=True)
        return sum(self.sums[0:3])


print(Puzzle(data).part_1())  # 69883
print(Puzzle(data).part_2())  # 207576
