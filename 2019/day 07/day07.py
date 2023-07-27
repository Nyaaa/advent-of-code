from copy import deepcopy
from itertools import permutations
from typing import Iterable

from tools import parsers, loader, intcode

init_pc = intcode.Intcode(parsers.lines(loader.get()))


def part_1():
    best_result = 0
    settings: Iterable[tuple[int]] = permutations(range(5), 5)
    for setting in settings:
        result = [0]
        for i in range(5):
            pc = deepcopy(init_pc)
            output = result[-1]
            sett = setting[i]
            res, _ = pc.run([sett, output])
            result.append(res)
        if result[-1] > best_result:
            best_result = result[-1]
    return best_result


def part_2():
    best_result = 0
    settings: Iterable[tuple[int]] = permutations(range(5, 10), 5)
    for setting in settings:
        setting = iter(setting)
        pcs = [deepcopy(init_pc) for _ in range(5)]
        pcs_done = []
        result = [0]
        while True:
            for i, pc in enumerate(pcs):
                output = result[-1]
                try:
                    sett = next(setting)
                    params = [sett, output]
                except StopIteration:
                    params = [output]
                res, done = pc.run(params)
                result.append(res)
                if done:
                    pcs_done.append(done)
            if len(pcs_done) == 5:
                break
        if result[-1] > best_result:
            best_result = result[-1]
    return best_result


print(part_1())  # 79723
print(part_2())  # 70602018
