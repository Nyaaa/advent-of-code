from collections import Counter
from itertools import combinations

from tools import loader, parsers


def part_1(data: list[str]) -> int:
    result = 0
    for line in data:
        if Counter(line.split()).most_common(1)[0][1] == 1:
            result += 1
    return result


def part_2(data: list[str]) -> int:
    result = 0
    for line in data:
        words = line.split()
        if Counter(words).most_common(1)[0][1] == 1 and \
                all(Counter(a) != Counter(b) for a, b in combinations(words, 2)):
            result += 1
    return result


print(part_1(parsers.lines(loader.get())))  # 325
print(part_2(parsers.lines(loader.get())))  # 119
