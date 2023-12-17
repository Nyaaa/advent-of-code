import re
from itertools import accumulate, chain, cycle, repeat

import numpy as np

from tools import loader, parsers


def race(data: list[str], time: int) -> tuple[int, int]:
    """
    >>> print(race(parsers.lines('test.txt'), time=1000))
    (1120, 689)"""
    reindeer = [tuple(map(int, re.findall(r'(\d+)', line))) for line in data]
    results = np.zeros((len(reindeer), time), dtype=int)
    for i, (speed, travel_time, rest_time) in enumerate(reindeer):
        fly = cycle(chain(repeat(speed, travel_time), repeat(0, rest_time)))
        results[i] = np.fromiter(accumulate(fly), dtype=int, count=time)

    scores = np.zeros((len(reindeer)), dtype=int)
    for window in results.T:
        scores[np.argwhere(window == max(window))] += 1

    return max(results[:, -1]), max(scores)


print(race(parsers.lines(loader.get()), time=2503))  # 2660, 1256
