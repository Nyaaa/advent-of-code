import re
from collections import defaultdict

from tools import loader, parsers

SAMPLE = {'children': 3,
          'cats': 7,
          'samoyeds': 2,
          'pomeranians': 3,
          'akitas': 0,
          'vizslas': 0,
          'goldfish': 5,
          'trees': 3,
          'cars': 2,
          'perfumes': 1
          }


def does_match(name: str, amount: int, mode: int) -> bool:
    match mode, name:
        case 2, 'cats' | 'trees':
            return amount >= SAMPLE[name]
        case 2, 'pomeranians' | 'goldfish':
            return amount <= SAMPLE[name]
        case _:
            return amount == SAMPLE[name]


def aunt_finder(data: list[str], part: int) -> int:
    aunts = defaultdict(dict)
    for i, aunt in enumerate(data, start=1):
        for item in re.finditer(r'(\w+): (\d+)', aunt):
            aunts[i][item.group(1)] = int(item.group(2))
    for aunt, items in aunts.items():
        for name, amount in items.items():
            if not does_match(name, amount, part):
                break
        else:
            return aunt
    raise ValueError('Aunt not found!')


print(aunt_finder(parsers.lines(loader.get()), part=1))  # 373
print(aunt_finder(parsers.lines(loader.get()), part=2))  # 260
