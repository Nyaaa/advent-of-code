import numpy as np

from tools import loader, parsers


def match_keys(data: list[list[str]]) -> int:
    """
    >>> print(match_keys(parsers.blocks('test.txt')))
    3"""
    locks = []
    keys = []
    for arr in data:
        array = np.array([[1 if i == '#' else 0 for i in line] for line in arr], dtype=int)
        group = locks if np.count_nonzero(array[0]) == 5 else keys
        group.append(array)
    return sum(np.all(np.add(lock, key) != 2) for lock in locks for key in keys)


print(match_keys(parsers.blocks(loader.get())))  # 3136
