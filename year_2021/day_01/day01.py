from tools import loader, parsers


class Sonar:
    def __init__(self, data: list[str]) -> None:
        self.data = [int(i) for i in data]

    def part_1(self) -> int:
        """
        >>> print(Sonar(parsers.lines('test.txt')).part_1())
        7"""
        return sum([1 for i in range(1, len(self.data)) if self.data[i - 1] < self.data[i]])

    def window(self, index: int) -> int:
        return sum([self.data[i] for i in (index, index - 1, index - 2)])

    def part_2(self) -> int:
        """
        >>> print(Sonar(parsers.lines('test.txt')).part_2())
        5"""
        return sum([1 for i in range(3, len(self.data)) if self.window(i - 1) < self.window(i)])


print(Sonar(parsers.lines(loader.get())).part_1())  # 1557
print(Sonar(parsers.lines(loader.get())).part_2())  # 1608
