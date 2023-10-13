from tools import loader, parsers

test = """forward 5
down 5
forward 8
up 3
down 8
forward 2
"""


class Sub:
    def __init__(self, data: list[str]) -> None:
        self.data = [[j[0], int(j[1])] for j in (i.split() for i in data)]
        self.depth = 0
        self.hor = 0
        self.aim = 0

    def part_1(self) -> int:
        """
        >>> print(Sub(parsers.inline_test(test)).part_1())
        150"""
        for command, val in self.data:
            if command == 'forward':
                self.hor += val
            elif command == 'down':
                self.depth += val
            elif command == 'up':
                self.depth -= val
        return self.depth * self.hor

    def part_2(self) -> int:
        """
        >>> print(Sub(parsers.inline_test(test)).part_2())
        900"""
        for command, val in self.data:
            if command == 'down':
                self.aim += val
            elif command == 'up':
                self.aim -= val
            elif command == 'forward':
                self.hor += val
                self.depth += self.aim * val
        return self.depth * self.hor


print(Sub(parsers.lines(loader.get())).part_1())  # 1728414
print(Sub(parsers.lines(loader.get())).part_2())  # 1765720035
