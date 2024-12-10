import numpy as np
from numba import njit
from numpy.typing import NDArray

from tools import loader, parsers

TEST = '2333133121414131402'


def defrag_part1(disk: NDArray) -> NDArray:
    right_ind = len(disk)
    free_space = 0
    while right_ind > free_space:
        right_ind -= 1
        right = disk[right_ind]
        if right == -1:
            continue
        disk[right_ind] = -1
        free_space = np.flatnonzero(disk == -1)[:1]
        disk[free_space] = right
    return disk


@njit
def defrag_part2(disk: NDArray) -> NDArray:
    right = disk[-1]
    while right != 0:
        right_file = np.flatnonzero(disk == right)
        right_ind = right_file[0]
        all_free = np.flatnonzero(disk[:right_ind] == -1)
        free_chunks = np.split(all_free, np.flatnonzero(np.diff(all_free) != 1) + 1)
        good_chunk = [i for i in free_chunks if len(i) >= right_file.size]
        if good_chunk:
            free_space = good_chunk[0][0]
            disk[right_ind: right_ind + right_file.size] = -1
            disk[free_space:free_space + right_file.size] = right
        right -= 1
    return disk


def defragment(data: str, part2: bool) -> int:
    """
    >>> print(defragment(TEST, part2=False))
    1928
    >>> print(defragment(TEST, part2=True))
    2858"""
    data_ = []
    counter = 0
    for i, val in enumerate(data):
        if i % 2 == 0:
            data_.extend(counter for _ in range(int(val)))
            counter += 1
        else:
            data_.extend(-1 for _ in range(int(val)))

    disk = np.asarray(data_, dtype=int)
    fn = defrag_part2 if part2 else defrag_part1
    disk = fn(disk)

    result = 0
    for i, val in enumerate(disk):
        if val == -1:
            continue
        result += i * val
    return result


print(defragment(parsers.string(loader.get()), part2=False))  # 6340197768906
print(defragment(parsers.string(loader.get()), part2=True))  # 6363913128533
