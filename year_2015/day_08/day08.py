from ast import literal_eval

from tools import loader, parsers


def part_1(data: list[str]) -> int:
    return sum(len(line) - len(literal_eval(line)) for line in data)


def part_2(data: list[str]) -> int:
    return sum(line.count('\\') + line.count('"') + 2 for line in data)


print(part_1(parsers.lines(loader.get())))  # 1371
print(part_2(parsers.lines(loader.get())))  # 2117
