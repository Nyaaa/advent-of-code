from numpy.typing import NDArray
from tools import parsers, loader
import numpy as np


TEST = """../.# => ##./#../...
.#./..#/### => #..#/..../..../#..#
"""


def all_rotations(tile: str) -> list[NDArray]:
    tile = np.array([list(1 if i == '#' else 0 for i in row) for row in tile.split('/')])
    rotations = []
    for _ in range(2):
        for _ in range(4):
            tile = np.rot90(tile)
            rotations.append(tile)
        tile = np.flip(tile, 0)
        rotations.append(tile)
    return rotations


def start(data: list[str], steps: int) -> int:
    """
    >>> print(start(parsers.inline_test(TEST), 2))
    12"""
    rules = {}
    for i in data:
        inp, out = i.split(' => ')
        out = np.array([list(1 if i == '#' else 0 for i in row) for row in out.split('/')])
        for rotation in all_rotations(inp):
            rules[rotation.tobytes()] = out
    image = np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]])

    for _ in range(steps):
        size = image.shape[0]
        n = 2 if size % 2 == 0 else 3
        cut_image = image.reshape(size // n, n, -1, n).swapaxes(1, 2).reshape(-1, n, n)
        new_image = [rules[tile.tobytes()] for i, tile in enumerate(cut_image)]
        num = int(np.sqrt(len(new_image)))
        image = np.block([new_image[i * num:i * num + num] for i in range(num)])
    return np.count_nonzero(image)


print(start(parsers.lines(loader.get()), 5))  # 208
print(start(parsers.lines(loader.get()), 18))  # 2480380
