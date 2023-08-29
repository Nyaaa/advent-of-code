from typing import NamedTuple
from tools import parsers, loader, intcode, common
import numpy as np
from more_itertools import split_at


def get_map():
    pc = intcode.Intcode(parsers.lines(loader.get()))
    pc.run()
    image = list(split_at(pc.logged_output, lambda x: x == 10))
    arr = np.asarray(image[:-2], dtype=int)
    arr[arr == 46] = 0
    arr[arr == 35] = 1
    return arr


def part_1():
    arr = get_map()
    params = []
    for i, val in np.ndenumerate(arr):
        if val == 1:
            adj = [i for _, i in common.get_adjacent(arr, i)]
            if len(adj) == 4 and all(adj):
                params.append(i[0] * i[1])
    return sum(params)


class Point(NamedTuple):
    row: int
    col: int

    def __add__(self, other):
        return Point(self.row + other.row, self.col + other.col)


DIRECTIONS = {'up': Point(-1, 0), 'down': Point(1, 0),
              'left': Point(0, -1), 'right': Point(0, 1)}
TURNS = {'up': ('left', 'right'),
         'right': ('up', 'down'),
         'down': ('right', 'left'),
         'left': ('down', 'up')}


def part_2():
    """Not a general solution, input-specific."""
    img = get_map()
    path = set(Point(i, j) for i, j in np.argwhere(img == 1))
    pos = Point(*np.argwhere(img == 94)[0])  # 94 = ^
    direction = 'up'
    track = []
    while True:
        new_pos = pos + DIRECTIONS[direction]
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
            new_pos = pos + DIRECTIONS[TURNS[direction][0]]
            if new_pos in path:
                direction = TURNS[direction][0]
                track.append('L')
            else:
                direction = TURNS[direction][1]
                track.append('R')

    program = ('A,C,A,B,A,C,B,C,B,C\n'
               'R,10,R,10,R,6,R,4\n'
               'R,4,L,4,L,10,L,10\n'
               'R,10,R,10,L,4\n'
               'n\n')
    compiled = list(map(ord, list(program)))
    pc = intcode.Intcode(parsers.lines(loader.get()))
    pc.data[0] = 2
    return pc.run(compiled)


print(part_1())  # 4112
print(part_2())  # 578918
