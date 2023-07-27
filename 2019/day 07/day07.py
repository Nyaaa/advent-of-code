from itertools import permutations
from typing import Iterable

from tools import parsers, loader, intcode

pc = intcode.Intcode(parsers.lines(loader.get()))
# pc = intcode.Intcode(["""3,31,3,32,1002,32,10,32,1001,31,-2,31,1007,31,0,33,
# 1002,33,7,33,1,33,31,31,1,32,31,31,4,31,99,0,0,0"""])


def part_1():
    best_result = 0
    settings: Iterable[tuple[int]] = permutations(range(5), 5)
    for setting in settings:
        result = [0]
        for i in range(5):
            output = result[-1]
            sett = setting[i]
            result.append(pc.run([sett, output]))
        if result[-1] > best_result:
            best_result = result[-1]
    return best_result


print(part_1())  # 79723


