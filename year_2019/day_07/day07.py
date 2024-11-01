from copy import deepcopy
from itertools import permutations

from tools import loader, parsers
from year_2019 import intcode

init_pc = intcode.Intcode(parsers.lines(loader.get()))


def part_1() -> int:
    best_result = 0
    for setting in permutations(range(5), 5):
        result = [0]
        for i in range(5):
            pc = deepcopy(init_pc)
            output = result[-1]
            sett = setting[i]
            res = pc.run([sett, output])
            result.append(res)
        best_result = max(result[-1], best_result)
    return best_result


def part_2() -> int:
    best_result = 0
    for setting in permutations(range(5, 10), 5):
        setting = iter(setting)
        pcs = [deepcopy(init_pc) for _ in range(5)]
        pcs_done = []
        result = [0]
        while True:
            for pc in pcs:
                output = result[-1]
                try:
                    sett = next(setting)
                    params = [sett, output]
                except StopIteration:
                    params = [output]
                res = pc.run(params)
                if pc.done:
                    result.append(res)
                    pcs_done.append(True)
                else:
                    result.extend(res)
            if len(pcs_done) == 5:
                break
        best_result = max(result[-1], best_result)
    return best_result


print(part_1())  # 79723
print(part_2())  # 70602018
