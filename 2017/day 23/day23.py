from collections import defaultdict

from tools import loader, parsers


class Coprocessor:
    def __init__(self, data: list[str]) -> None:
        self.program = data
        self.mem = defaultdict(int)

    def get_value(self, value: str) -> int:
        try:
            return int(value)
        except ValueError:
            return self.mem[value]

    def part_1(self) -> int:
        index = 0
        while index < 11:
            match self.program[index].split():
                case 'set', x, y:
                    self.mem[x] = self.get_value(y)
                case 'sub', x, y:
                    self.mem[x] -= self.get_value(y)
                case 'mul', x, y:
                    self.mem[x] *= self.get_value(y)
                case 'jnz', x, y if self.get_value(x) != 0:
                    index += self.get_value(y)
                    continue
            index += 1
        return (self.mem['b'] - self.mem['e']) * (self.mem['b'] - self.mem['d'])

    @staticmethod
    def is_prime(num: int) -> bool:
        return all(num % i != 0 for i in range(2, num))

    def part_2(self) -> int:
        self.mem['a'] = 1
        self.part_1()
        return sum(not self.is_prime(num) for num in range(self.mem['b'], self.mem['c'] + 1, 17))


print(Coprocessor(parsers.lines(loader.get())).part_1())  # 9409
print(Coprocessor(parsers.lines(loader.get())).part_2())  # 913
