import re

from tools import loader, parsers


def passes_condition(condition: tuple[str | int], part: dict[str, int]) -> str | bool:
    part_value = part.get(condition[0])
    if not part_value:
        return condition[0]
    if part_value < condition[2] if condition[1] == '<' else part_value > condition[2]:
        return condition[3]
    return False


def sort_parts(data: list[list[str]]) -> int:
    """
    >>> print(sort_parts(parsers.blocks('test.txt')))
    19114"""
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
    results = {'A': [], 'R': []}
    for part in parts:
        current_rule = rules['in']
        while current_rule:
            for cond in current_rule:
                p = passes_condition(cond, part)
                if not p:
                    continue
                if p in 'AR':
                    results[p].append(part)
                    current_rule = None
                    break
                if p in rules:
                    current_rule = rules[p]
                    break
    part1 = [sum(i.values()) for i in results['A']]
    return sum(part1)


print(sort_parts(parsers.blocks(loader.get())))  # 446517

