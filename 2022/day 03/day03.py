import string
from tools import parsers, loader

letters = list(string.ascii_letters)
numbers = list(range(1, 53))
priority = dict(zip(letters, numbers))
test = ['vJrwpWtwJgWrhcsFMMfFFhFp',
        'jqHRNqRjqzjGDLGLrsFMfFZSrLrFZsSL',
        'PmmdzqPrVvPwwTWBwg',
        'wMqvLMZHhHMvwLHjbvcjnnSBnvTQFn',
        'ttgJtRGJQctTZtZT',
        'CrZsJsPPZsGzwwsLwLmpwMDw']


def common(one, *rest):
    return list(set(one).intersection(*rest))


def part_1(data):
    """test part 1:
    >>> print(part_1(test))
    157"""
    value = 0
    for rucksack in data:
        mid = len(rucksack) // 2
        pocket1 = list(rucksack[:mid])
        pocket2 = list(rucksack[mid:])
        item = common(pocket1, pocket2)
        value += priority[item[0]]
    return value


def part_2(data):
    """test part 2:
    >>> print(part_2(test))
    70"""
    value = 0
    for i in range(0, len(data), 3):
        a, b, c = data[i], data[i + 1], data[i + 2]
        item = common(a, b, c)
        value += priority[item[0]]
    return value


print(part_1(parsers.lines(loader.get())))  # 7674
print(part_2(parsers.lines(loader.get())))  # 2805
