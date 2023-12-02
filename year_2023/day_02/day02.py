import re
from dataclasses import dataclass

from tools import loader, parsers


@dataclass
class Bag:
    ID: int
    red: int = 0
    green: int = 0
    blue: int = 0

    def is_valid(self) -> bool:
        return self.red <= 12 and self.green <= 13 and self.blue <= 14

    def add_cube(self, colour: str, amount: int) -> None:
        setattr(self, colour, max(getattr(self, colour), amount))

    def power(self) -> int:
        return self.red * self.green * self.blue


def cubes(data: list[str]) -> tuple[int, int]:
    """
    >>> print(cubes(parsers.lines('test.txt')))
    (8, 2286)"""
    bags = []
    for line in data:
        bag = Bag(ID=int(re.findall(r'\d+', line)[0]))
        for cube in re.finditer(r'(\d+) (\w+)', line):
            bag.add_cube(cube.group(2), int(cube.group(1)))
        bags.append(bag)
    part1 = sum(bag.ID for bag in bags if bag.is_valid())
    part2 = sum(bag.power() for bag in bags)
    return part1, part2


print(cubes(parsers.lines(loader.get())))  # 2369, 66363
