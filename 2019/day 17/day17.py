from tools import parsers, loader, intcode, common
import numpy as np
from more_itertools import split_at


def part_1():
    pc = intcode.Intcode(parsers.lines(loader.get()))
    pc.run()
    image = list(split_at(pc.logged_output, lambda x: x == 10))
    arr = np.asarray(image[:-2], dtype=int)
    arr[arr == 46] = 0
    arr[arr == 35] = 1
    params = []
    for i, val in np.ndenumerate(arr):
        if val == 1:
            adj = [i for _, i in common.get_adjacent(arr, i)]
            if len(adj) == 4 and all(adj):
                params.append(i[0] * i[1])
    return sum(params)


def part_2():
    pc = intcode.Intcode(parsers.lines(loader.get()))
    pc.run()
    image = ''.join(chr(i) for i in pc.logged_output)
    return image


print(part_1())  # 4112
print(part_2())
