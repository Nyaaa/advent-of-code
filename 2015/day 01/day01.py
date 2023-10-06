from tools import loader, parsers


def part_1(data: str) -> int:
    return data.count('(') - data.count(')')


def part_2(data: str) -> int:
    floor = 0
    for i, val in enumerate(data):
        floor += 1 if val == '(' else -1
        if floor < 0:
            return i + 1
    return 0


print(part_1(parsers.string(loader.get())))  # 232
print(part_2(parsers.string(loader.get())))  # 1783


