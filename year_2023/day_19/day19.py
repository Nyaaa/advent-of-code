import re
from collections import deque
from collections.abc import Generator
from math import prod

from tools import loader, parsers


def parse_input(data: list[list[str]]) -> tuple[list[dict[str, int]], dict[str, list[tuple]]]:
    parts = [{i[0]: int(i[1]) for i in re.findall(r'(\w)=(\d+)', line)} for line in data[1]]
    rules = {}
    for line in data[0]:
        name, _rules = line.split('{')
        rules[name] = []
        conditions = _rules[:-1].split(',')
        for c in conditions:
            for match in re.finditer(r'(.)(.)(\d+):(\w+),?(\w+)?|(\w+)', c):
                rules[name].append(
                    tuple(int(i) if i.isdigit() else i for i in match.groups() if i))
    return parts, rules


def check_condition(condition: tuple[str | int], part: dict[str, int]) -> str | bool:
    part_value = part.get(condition[0])
    if not part_value:
        return condition[0]
    if part_value < condition[2] if condition[1] == '<' else part_value > condition[2]:
        return condition[3]
    return False


def part_1(data: list[list[str]]) -> int:
    """
    >>> print(part_1(parsers.blocks('test.txt')))
    19114"""
    parts, rules = parse_input(data)
    results = {'A': [], 'R': []}
    for part in parts:
        current_rule = rules['in']
        while current_rule:
            for cond in current_rule:
                p = check_condition(cond, part)
                if not p:
                    continue
                if p in 'AR':
                    results[p].append(part)
                    current_rule = None
                    break
                if p in rules:
                    current_rule = rules[p]
                    break
    return sum(sum(i.values()) for i in results['A'])


def split_range(part: dict[str, tuple[int, int]],
                rule: list[tuple]) -> Generator[tuple[dict[str, tuple[int, int]], str]]:
    for condition in rule:
        if len(condition) == 1:
            yield part, condition[0]
            break
        low, high = part.get(condition[0])
        low_part = part.copy()
        if condition[1] == '<':
            part[condition[0]] = (condition[2], high)
            low_part[condition[0]] = (low, condition[2] - 1)
            yield low_part, condition[3]
        else:
            part[condition[0]] = (condition[2] + 1, high)
            low_part[condition[0]] = (low, condition[2])
            yield part, condition[3]
            part = low_part


def part_2(data: list[list[str]]) -> int:
    """
    >>> print(part_2(parsers.blocks('test.txt')))
    167409079868000"""
    _, rules = parse_input(data)
    part = {i: (1, 4000) for i in 'xmas'}
    queue = deque([(part, rules['in'])])
    results = {'A': [], 'R': []}
    while queue:
        part, rule = queue.pop()
        for new_range, next_rule in split_range(part, rule):
            if next_rule in 'AR':
                results[next_rule].append(new_range)
            else:
                queue.append((new_range, rules[next_rule]))
    out = 0
    for result in [i.values() for i in results['A']]:
        out += prod([high - low + 1 for (low, high) in result])
    return out


print(part_1(parsers.blocks(loader.get())))  # 446517
print(part_2(parsers.blocks(loader.get())))  # 130090458884662
