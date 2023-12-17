import re
from itertools import starmap

import numpy as np
from more_itertools import split_at
from numpy.typing import NDArray

from tools import common, loader, parsers
from year_2019 import intcode


def get_map() -> NDArray:
    pc = intcode.Intcode(parsers.lines(loader.get()))
    pc.run()
    image = list(split_at(pc.logged_output, lambda x: x == 10))
    arr = np.asarray(image[:-2], dtype=int)
    arr[arr == 46] = 0
    arr[arr == 35] = 1
    return arr


def part_1() -> int:
    arr = get_map()
    params = []
    for i, val in np.ndenumerate(arr):
        if val == 1:
            adj = [i for _, i in common.get_adjacent(arr, i)]
            if len(adj) == 4 and all(adj):
                params.append(i[0] * i[1])
    return sum(params)


def get_path() -> str:
    img = get_map()
    path = set(starmap(complex, np.argwhere(img == 1)))
    pos = complex(*np.argwhere(img == 94)[0])  # 94 = ^
    direction = -1
    track = []
    while True:
        new_pos = pos + direction
        if new_pos in path:
            pos = new_pos
            if isinstance(track[-1], int):
                track[-1] += 1
            else:
                track.append(1)
        else:
            if len(track) > 2 and isinstance(track[-1], str):
                track.pop()
                break
            new_pos = pos + direction * 1j
            if new_pos in path:
                direction *= 1j
                track.append('L')
            else:
                direction *= -1j
                track.append('R')
    return ','.join(str(i) for i in track)


def part_2() -> int:
    path = get_path() + ','
    pattern = re.compile(r'^(.{1,20})\1*?(.{1,20})(?:\1|\2)*?(.{1,20})(?:\1|\2|\3)*$')
    chunks = dict(zip(re.match(pattern, path).groups(), ['A', 'B', 'C'], strict=True))
    program = ''
    for chunk, letter in chunks.items():
        path = re.sub(chunk, letter, path)
        program += f'{chunk[:-1]}\n'
    program = f"{','.join(path)}\n{program}\n\n"
    compiled = list(map(ord, list(program)))
    pc = intcode.Intcode(parsers.lines(loader.get()))
    pc.data[0] = 2
    return pc.run(compiled)


print(part_1())  # 4112
print(part_2())  # 578918
