import re

from more_itertools import transpose

from tools import loader, parsers

TEST = """Time:      7  15   30
Distance:  9  40  200
"""


def race(data: list[str], part2: bool = False) -> int:
    """
    >>> print(race(parsers.inline_test(TEST), part2=False))
    288
    >>> print(race(parsers.inline_test(TEST), part2=True))
    71503"""
    records = list(transpose(list(map(int, re.findall(r'\d+', i.split(':')[1]))) for i in data))
    if part2:
        records = [tuple(int(re.sub(r'\s+', '', i.split(':')[1])) for i in data)]
    wins = 1
    for time, record in records:
        wins *= sum(record < (time - hold) * hold for hold in range(1, time + 1))
    return wins


print(race(parsers.lines(loader.get()), part2=False))  # 138915
print(race(parsers.lines(loader.get()), part2=True))  # 27340847
