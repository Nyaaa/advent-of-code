import itertools
import re
from collections import Counter, defaultdict

from tools import loader, parsers


class Allergens:
    def __init__(self, data: list[str]) -> None:
        self.all_ingredients = []
        self.probable_allergens: defaultdict[str, list | set] = defaultdict(list)
        self.identified_allergens = {}
        for line in data:
            items = re.findall(r'\w+', line)
            ingredients, allergens = (list(value) for key, value in
                                      itertools.groupby(items, lambda e: e == 'contains')
                                      if not key)
            for i in allergens:
                self.probable_allergens[i].append(ingredients)
            self.all_ingredients.extend(ingredients)
        self.identify_allergens()

    def identify_allergens(self) -> None:
        for allergen, potentials in self.probable_allergens.items():
            self.probable_allergens[allergen] = set.intersection(*[set(i) for i in potentials])

        while self.probable_allergens.keys() != self.identified_allergens.keys():
            for allergen, potentials in self.probable_allergens.items():
                potentials = {i for i in potentials if i not in self.identified_allergens.values()}
                if len(potentials) == 1:
                    self.identified_allergens[allergen] = potentials.pop()

    def part_1(self) -> int:
        """
        >>> print(Allergens(parsers.lines('test.txt')).part_1())
        5"""
        safe = set(self.all_ingredients) - set(self.identified_allergens.values())
        counts = Counter(self.all_ingredients)
        result = 0
        for i in safe:
            result += counts[i]
        return result

    def part_2(self) -> str:
        """
        >>> print(Allergens(parsers.lines('test.txt')).part_2())
        mxmxvkd,sqjhc,fvjkl"""
        sorted_dict = dict(sorted(self.identified_allergens.items()))
        return ','.join(sorted_dict.values())


print(Allergens(parsers.lines(loader.get())).part_1())  # 2176
print(Allergens(parsers.lines(loader.get())).part_2())  # lvv,xblchx,tr,gzvsg,jlsqx,fnntr,pmz,csqc
