from collections import defaultdict

from tools import parsers, loader
import re
import itertools


class Allergens:
    def __init__(self, data):
        self.allergens = set()
        self.ingredients = list()
        self.probable_allergens = defaultdict(list)
        for x in data:
            a = re.findall(r'\w+', x)
            ingredients, allergens = [list(value) for key, value in
                                      itertools.groupby(a, lambda e: e == 'contains')
                                      if not key]
            for i in allergens:
                self.probable_allergens[i].append(ingredients)
            self.ingredients.extend(ingredients)

    def part_1(self):
        """"
        >>> print(Allergens(parsers.lines('test.txt')).part_1())
        5"""
        for allergen, potentials in self.probable_allergens.items():
            pot = [set(i) for i in potentials]
            for i in potentials[1:]:
                common = pot[0].intersection(i)
                self.allergens.update(common)
            if len(pot) == 1:
                self.allergens.update(pot[0])
        return len([i for i in self.ingredients if i not in self.allergens])


print(Allergens(parsers.lines(loader.get())).part_1())  # 268 too low

