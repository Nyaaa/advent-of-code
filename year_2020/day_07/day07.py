import re
from collections import deque

from tools import loader, parsers


def parse(data: list[str]) -> dict:
    """
    >>> print(parse(['light red bags contain 1 bright white bag, 2 muted yellow bags.']))
    {'light red': {'bright white': 1, 'muted yellow': 2}}"""
    out = {}
    for line in data:
        parent = re.findall(r'^(\w+\s\w+)', line)[0]
        out[parent] = {}
        for child in re.findall(r'(\d+)\s(\w+\s\w+)\sbag', line):
            out[parent][child[1]] = int(child[0])
    return out


def part_1(bags: dict) -> int:
    """
    >>> print(part_1(parse(parsers.lines('test.txt'))))
    4"""
    start = deque(['shiny gold'])
    results = set()
    while start:
        current_bag = start.popleft()
        for i in bags:
            if bags.get(i).get(current_bag):
                results.add(i)
                start.append(i)
    return len(results)


def part_2(all_bags: dict, bag: str = 'shiny gold') -> int:
    """
    >>> print(part_2(parse(parsers.lines('test.txt'))))
    32"""
    inner = all_bags[bag]
    result = sum(inner.values())
    if inner:
        for inner_bag, amount in inner.items():
            result += part_2(all_bags, inner_bag) * amount
    return result


print(part_1(parse(parsers.lines(loader.get()))))  # 233
print(part_2(parse(parsers.lines(loader.get()))))  # 421550
