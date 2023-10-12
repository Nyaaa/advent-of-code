from tools import parsers, loader
from numba import njit


@njit
def generate_map(data: str, rows: int) -> int:
    """
    >>> print(generate_map('.^^.^.^^^^', 10))
    38"""
    tiles = [False if i == '^' else True for i in data]
    zeros = 0

    for _ in range(rows):
        zeros += sum(tiles)
        row = [True] + tiles.copy() + [True]
        for i, (a, b) in enumerate(zip(row, row[2:])):
            if (a and not b) or (not a and b):
                tiles[i] = False
            else:
                tiles[i] = True
    return zeros


print(generate_map(parsers.string(loader.get()), 40))  # 1963
print(generate_map(parsers.string(loader.get()), 400_000))  # 20009568
