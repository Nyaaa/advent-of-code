import string

from tools import loader, parsers

letters = list(string.ascii_letters)
numbers = list(range(1, 53))
priority = dict(zip(letters, numbers))
test = ['vJrwpWtwJgWrhcsFMMfFFhFp',
        'jqHRNqRjqzjGDLGLrsFMfFZSrLrFZsSL',
        'PmmdzqPrVvPwwTWBwg',
        'wMqvLMZHhHMvwLHjbvcjnnSBnvTQFn',
        'ttgJtRGJQctTZtZT',
        'CrZsJsPPZsGzwwsLwLmpwMDw']


def common(one: str, *rest: str) -> list:
    return list(set(one).intersection(*rest))


def part_1(data: list[str]) -> int:
    """test part 1:
    >>> print(part_1(test))
    157"""
    value = 0
    for rucksack in data:
        mid = len(rucksack) // 2
        item = common(rucksack[:mid], rucksack[mid:])
        value += priority[item[0]]
    return value


def part_2(data: list[str]) -> int:
    """test part 2:
    >>> print(part_2(test))
    70"""
    value = 0
    for i in range(0, len(data), 3):
        item = common(data[i], data[i + 1], data[i + 2])
        value += priority[item[0]]
    return value


print(part_1(parsers.lines(loader.get())))  # 7674
print(part_2(parsers.lines(loader.get())))  # 2805
