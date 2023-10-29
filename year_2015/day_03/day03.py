from collections import defaultdict
from dataclasses import dataclass
from itertools import cycle

from tools import loader, parsers

DIRS = {'^': -1, '>': 1j, 'v': 1, '<': -1j}


@dataclass
class Actor:
    position = 0+0j


def start(data: str, part_2: bool) -> int:
    houses = defaultdict(int)
    santa, robot = Actor(), Actor()
    houses[santa.position] += 2
    actors = cycle([santa, robot])
    for i in data:
        unit = next(actors) if part_2 else santa
        unit.position += DIRS[i]
        houses[unit.position] += 1
    return sum(i >= 1 for i in houses.values())


print(start(parsers.string(loader.get()), part_2=False))  # 2081
print(start(parsers.string(loader.get()), part_2=True))  # 2341
