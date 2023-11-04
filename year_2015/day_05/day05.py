import re

from tools import loader, parsers


def part_1(data: list[str]) -> int:
    """
    >>> print(part_1(['ugknbfddgicrmopn']))
    1
    >>> print(part_1(['aaa']))
    1
    >>> print(part_1(['jchzalrnumimnmhp']))
    0
    >>> print(part_1(['haegwjzuvuyypxyu']))
    0
    >>> print(part_1(['dvszwmarrgswjxmb']))
    0"""
    condition = re.compile(r'(?=.*(.)\1)(?=(.*[aeiou]){3})(?!(.*(ab|cd|pq|xy)))')
    return sum(bool(re.match(condition, string)) for string in data)


def part_2(data: list[str]) -> int:
    """
    >>> print(part_2(['qjhvhtzxzqqjkmpb']))
    1
    >>> print(part_2(['xxyxx']))
    1
    >>> print(part_2(['uurcxstgmygtbstg']))
    0
    >>> print(part_2(['ieodomkazucvgmuy']))
    0"""
    condition = re.compile(r'(?=.*(..).*\1)(?=.*(.).\2)')
    return sum(bool(re.match(condition, string)) for string in data)


print(part_1(parsers.lines(loader.get())))  # 236
print(part_2(parsers.lines(loader.get())))  # 51
