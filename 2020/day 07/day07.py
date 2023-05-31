import re
from collections import deque
from tools import parsers, loader


def parse(data: list[str]):
    """
    >>> print(parse(['light red bags contain 1 bright white bag, 2 muted yellow bags.']))
    {'light red': {'bright white': '1', 'muted yellow': '2'}}"""
    out = dict()
    for line in data:
        parent = re.findall(r'^(\w+\s\w+)', line)[0]
        out[parent] = dict()
        for child in re.findall(r'(\d+)\s(\w+\s\w+)\sbag', line):
            out[parent][child[1]] = child[0]
    return out


def part_1(bags: dict):
    """
    >>> print(part_1(test_data))
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


test_data = parse(parsers.lines('test.txt'))
input_data = parse(parsers.lines(loader.get()))

print(part_1(input_data))  # 233
