from numba import njit

from tools import loader, parsers


@njit
def generate_map(data: str, rows: int) -> int:
    """
    >>> print(generate_map('.^^.^.^^^^', 10))
    38"""
    tiles = [i != '^' for i in data]
    zeros = 0

    for _ in range(rows):
        zeros += sum(tiles)
        row = [True] + tiles.copy() + [True]
        for i, (a, b) in enumerate(zip(row, row[2:])):
            tiles[i] = not ((a and not b) or (not a and b))
    return zeros


print(generate_map(parsers.string(loader.get()), 40))  # 1963
print(generate_map(parsers.string(loader.get()), 400_000))  # 20009568
