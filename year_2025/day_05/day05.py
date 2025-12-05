import operator

from tools import loader, parsers


def part1(data: list[list[str]]) -> int:
    """
    >>> print(part1(parsers.blocks('test.txt')))
    3
    """
    ids = [range(start, end + 1) for item in data[0] for start, end in [map(int, item.split('-'))]]
    return sum(any(int(i) in r for r in ids) for i in data[1])


def part2(data: list[list[str]]) -> int:
    """
    >>> print(part2(parsers.blocks('test.txt')))
    14
    """
    ids = [[start, end + 1] for item in data[0] for start, end in [map(int, item.split('-'))]]
    ids.sort(key=operator.itemgetter(0))
    merged = [ids[0]]

    for current in ids[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            last[1] = max(last[1], current[1])
        else:
            merged.append(current)

    return sum(i[1] - i[0] for i in merged)


print(part1(parsers.blocks(loader.get())))  # 735
print(part2(parsers.blocks(loader.get())))  # 344306344403172
