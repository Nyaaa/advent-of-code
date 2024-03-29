from copy import deepcopy

from tools import loader, parsers
from year_2019 import intcode


def part_1() -> int:
    pc = deepcopy(init_pc)
    pc.data[1] = 12
    pc.data[2] = 2
    return pc.run()[0]


def part_2() -> int:
    for i in range(100):
        for j in range(100):
            pc = deepcopy(init_pc)
            pc.data[1] = i
            pc.data[2] = j
            if pc.run()[0] == 19690720:
                return 100 * i + j
    raise ValueError


init_pc = intcode.Intcode(parsers.lines(loader.get()))
print(part_1())  # 5290681
print(part_2())  # 5741
