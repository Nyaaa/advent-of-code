import sys
from collections import deque, defaultdict
from math import ceil
from tools import parsers, loader
import re


class Factory:
    def __init__(self, data: list[str]):
        self.recipes = {}
        for line in data:
            elements = re.findall(r'(\d+)\s+(\w+)', line)
            self.recipes[elements[-1][1]] = (int(elements[-1][0]), {k: int(v) for v, k in elements[:-1]})

    def produce(self, requirement: int):
        production = deque([('FUEL', requirement)])
        inventory = defaultdict(int)

        while production:
            element, amount = production.popleft()
            if element == 'ORE':
                inventory['ORE'] += amount
                continue
            if amount <= inventory[element]:
                inventory[element] -= amount
            else:
                request_num = amount - inventory[element]
                output_num, recipes = self.recipes[element]
                multiplier = ceil(request_num / output_num)
                production.extendleft(((i[0], i[1] * multiplier) for i in recipes.items()))
                inventory[element] = multiplier * output_num - request_num
        return inventory['ORE']

    def part_1(self):
        """
        >>> print(Factory(parsers.lines('test.txt')).part_1())
        2210736"""
        return self.produce(1)

    def part_2(self):
        """Binary search
        >>> print(Factory(parsers.lines('test.txt')).part_2())
        460664"""
        left = 0
        right = sys.maxsize
        while right != left:
            middle = (left + right) // 2
            if self.produce(middle) > 1000000000000:
                right = middle - 1
            else:
                left = middle + 1
        return left


print(Factory(parsers.lines(loader.get())).part_1())  # 899155
print(Factory(parsers.lines(loader.get())).part_2())  # 2390226
