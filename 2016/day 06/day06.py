from collections import Counter

import numpy as np

from tools import loader, parsers


def start(data: list[str]) -> tuple[str, str]:
    """
    >>> print(start(parsers.lines('test.txt')))
    ('easter', 'advent')"""
    arr = np.asarray([list(line) for line in data])
    part1 = part2 = ''
    for i in range(arr.shape[1]):
        counts = Counter(arr[:, i]).most_common()
        part1 += counts[0][0]
        part2 += counts[-1][0]
    return part1, part2


print(start(parsers.lines(loader.get())))  # kjxfwkdh, xrwcsnps
