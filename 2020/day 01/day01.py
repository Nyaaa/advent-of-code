from tools import parsers, loader
from itertools import combinations
from math import prod


TEST = """1721
979
366
299
675
1456
"""


def find_expense(data: list, sums_num: int) -> int:
    """test part 1:
    >>> print(find_expense(parsers.inline_test(TEST), 2))
    514579

    test part 2:
    >>> print(find_expense(parsers.inline_test(TEST), 3))
    241861950
    """
    combs = combinations((int(i) for i in data), sums_num)
    return prod(*(i for i in combs if sum(i) == 2020))


print(find_expense(parsers.lines(loader.get()), 2))  # 910539
print(find_expense(parsers.lines(loader.get()), 3))  # 116724144
