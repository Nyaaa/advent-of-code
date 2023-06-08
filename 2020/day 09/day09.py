from itertools import combinations
from tools import parsers, loader


def part_1(data: list, chunk: int = 25, part2: bool = False) -> int:
    """ test part 1:
    >>> print(part_1(parsers.lines('test.txt'), chunk=5))
    127

    test part 2:
    >>> print(part_1(parsers.lines('test.txt'), chunk=5, part2=True))
    62"""
    data = [int(i) for i in data]
    value = None
    for i, val in enumerate(data):
        if i < chunk:
            continue
        preamble = data[i-chunk:i]
        if val not in (sum(i) for i in combinations(preamble, 2)):
            value = val
    if part2:
        lengths = []
        for i, v in enumerate(data):
            counter = 0
            _sum = 0
            while _sum <= value:
                counter += 1
                sequence = data[i:i + counter]
                _sum = sum(sequence)
                if _sum == value:
                    lengths.append(sequence)
        longest = sorted(max(lengths, key=len))
        return longest[0] + longest[-1]
    return value


print(part_1(parsers.lines(loader.get())))  # 177777905
print(part_1(parsers.lines(loader.get()), part2=True))  # 23463012
