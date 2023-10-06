from tools import loader, parsers


class Puzzle:
    def __init__(self, data: list[list[str]]) -> None:
        self.sums = [sum(int(i) for i in block) for block in data]

    def part_1(self) -> int:
        """test part 1:
        >>> print(Puzzle(parsers.blocks('test.txt')).part_1())
        24000"""
        return max(self.sums)

    def part_2(self) -> int:
        """test part 2:
        >>> print(Puzzle(parsers.blocks('test.txt')).part_2())
        45000"""
        self.sums.sort(reverse=True)
        return sum(self.sums[0:3])


print(Puzzle(parsers.blocks(loader.get())).part_1())  # 69883
print(Puzzle(parsers.blocks(loader.get())).part_2())  # 207576
