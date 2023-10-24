import importlib
import re
from inspect import getmembers, isfunction

from tools import loader, parsers


class Program:
    def __init__(self, data: list[str]) -> None:
        self.ip_register = int(re.findall(r'\d+', data[0])[0])
        self.instructions = {i: (v.split()[0], (0, *map(int, re.findall(r'\d+', v))))
                             for i, v in enumerate(data[1:])}
        self.ops = dict(getmembers(
            importlib.import_module('year_2018.day_16.operators'), isfunction))
        self.register = [0, 0, 0, 0, 0, 0]

    def part_1(self) -> int:
        while True:
            ip = self.register[self.ip_register]
            op, instruction = self.instructions[ip]
            self.register[self.ip_register] = ip
            register = self.ops[op](instruction, self.register)

            # optimized calculation
            if op == 'seti' and register[self.ip_register] == 0:  # start of a loop
                number = register[2]
                return sum(i for i in range(1, number + 1) if number % i == 0)

            ip = register[self.ip_register] + 1
            if not 0 <= ip < len(self.instructions):
                break
            register[self.ip_register] = ip
        return register[0]  # non-optimized calculation

    def part_2(self) -> int:
        self.register[0] = 1
        return self.part_1()


print(Program(parsers.lines(loader.get())).part_1())  # 1620
print(Program(parsers.lines(loader.get())).part_2())  # 15827082
