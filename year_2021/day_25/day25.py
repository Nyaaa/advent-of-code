import numpy as np

from tools import loader, parsers


def simulate(data: list[str]) -> int:
    """
    >>> print(simulate(parsers.lines('test.txt')))
    58"""
    def move(axis: int) -> bool:
        all_cucumbers = (arr != 2)
        cucumbers_to_move = (arr == axis)
        after_moving = np.roll(cucumbers_to_move, axis=axis, shift=1)
        new_positions = (after_moving & ~all_cucumbers)
        old_positions = np.roll(new_positions, axis=axis, shift=-1)
        arr[old_positions] = 2
        arr[new_positions] = axis
        return np.any(new_positions)

    arr = np.genfromtxt(data, delimiter=1, dtype=str)
    arr[arr == '>'] = 1
    arr[arr == 'v'] = 0
    arr[arr == '.'] = 2
    arr = arr.astype(int)
    moved = True
    step = 0
    while moved:
        moved = any([move(1), move(0)])
        step += 1
    return step


print(simulate(parsers.lines(loader.get())))  # 400
