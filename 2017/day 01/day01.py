from tools import parsers, loader


def part_1(data: str):
    """
    >>> print(part_1('1122'))
    3"""
    return sum(int(val) for i, val in enumerate(data) if val == data[(i + 1) % len(data)])


def part_2(data: str):
    """
    >>> print(part_2('123425'))
    4"""
    return sum(int(val) for i, val in enumerate(data) if val == data[(i + len(data) // 2) % len(data)])


print(part_1(parsers.string(loader.get())))  # 1228
print(part_2(parsers.string(loader.get())))  # 1238
