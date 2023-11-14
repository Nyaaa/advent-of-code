import re
from collections import Counter
from itertools import combinations_with_replacement
from math import prod

from tools import loader, parsers


def find_recipe(data: list[str]) -> tuple[int, int]:
    """
    >>> print(find_recipe(parsers.lines('test.txt')))
    (62842880, 57600000)"""
    ingredients = {tuple(map(int, re.findall(r'-?\d+', line))) for line in data}
    options = combinations_with_replacement(ingredients, 100)
    part1 = 0
    part2 = 0
    for recipe in options:
        items = Counter(recipe)
        score = prod(max(0, sum(item[index] * items[item] for item in items))
                     for index in range(4))
        part1 = max(score, part1)
        if sum(item[-1] * items[item] for item in items) == 500:
            part2 = max(score, part2)
    return part1, part2


print(find_recipe(parsers.lines(loader.get())))  # 222870, 117936
