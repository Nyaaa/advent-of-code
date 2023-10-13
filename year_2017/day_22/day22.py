from tools import loader, parsers

TEST = """..#
#..
...
"""


class Virus:
    def __init__(self, data: list[str]) -> None:
        self.infected = {}  # row, column
        for i, row in enumerate(data):
            for j, cell in enumerate(row):
                if cell == '#':
                    self.infected[complex(i, j)] = 'I'
        self.current_node = complex(len(data) // 2, len(data[0]) // 2)
        self.direction = -1
        self.count = 0

    def part_1(self) -> int:
        """
        >>> print(Virus(parsers.inline_test(TEST)).part_1())
        5587"""
        for _ in range(10_000):
            if self.current_node in self.infected:
                self.direction *= -1j
                del self.infected[self.current_node]
            else:
                self.direction *= 1j
                self.infected[self.current_node] = 'I'
                self.count += 1
            self.current_node += self.direction
        return self.count

    def part_2(self) -> int:
        """
        >>> print(Virus(parsers.inline_test(TEST)).part_2())
        2511944"""
        for _ in range(10_000_000):
            match self.infected.get(self.current_node):
                case 'I':
                    self.direction *= -1j
                    self.infected[self.current_node] = 'F'
                case 'W':
                    self.infected[self.current_node] = 'I'
                    self.count += 1
                case 'F':
                    self.direction = -self.direction
                    self.infected[self.current_node] = 'C'
                case _:
                    self.direction *= 1j
                    self.infected[self.current_node] = 'W'
            self.current_node += self.direction
        return self.count


print(Virus(parsers.lines(loader.get())).part_1())  # 5552
print(Virus(parsers.lines(loader.get())).part_2())  # 2511527
