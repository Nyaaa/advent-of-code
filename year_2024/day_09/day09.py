from collections import deque
from collections.abc import Iterable

from tools import loader, parsers

TEST = '2333133121414131402'


def prep_data(data: str) -> deque[int]:
    disk = deque([])
    counter = 0
    for i, val in enumerate(data):
        if i % 2 == 0:
            disk.extend(counter for _ in range(int(val)))
            counter += 1
        else:
            disk.extend(-1 for _ in range(int(val)))
    return disk


def get_result(done: Iterable[int]) -> int:
    result = 0
    for i, val in enumerate(done):
        if val == -1:
            continue
        result += i * int(val)
    return result


def part1(data: str) -> int:
    """
    >>> print(part1(TEST))
    1928"""
    done = []
    disk = prep_data(data)

    while disk:
        while disk:
            left = disk.popleft()
            if left != -1:
                done.append(left)
            else:
                break

        if not disk:
            break

        right = disk.pop()
        disk.appendleft(right)

    return get_result(done)


def part2(data: str) -> int:
    """
    >>> print(part2(TEST))
    2858"""
    disk = list(prep_data(data))

    right = None
    right_ind = None
    while right_ind != 0:
        if not right:
            right = disk[-1]
            right_ind = disk.index(right)
        right_len = disk.count(right)

        indices = []
        left_ind = disk.index(-1)
        for i, val in enumerate(disk[left_ind:right_ind], start=left_ind):
            if val == -1:
                indices.append(i)
            else:
                indices = []
                continue

            if len(indices) >= right_len and indices[0] < right_ind:
                for i_right in range(right_len):
                    disk[right_ind + i_right] = -1
                for index in indices:
                    disk[index] = right
                right -= 1
                right_ind = disk.index(right)
                break
        else:
            last_moved = right
            for val in reversed(disk[:right_ind]):
                if val < last_moved and val != -1:
                    right = val
                    right_ind = disk.index(right)
                    break

    return get_result(disk)


print(part1(parsers.string(loader.get())))  # 6340197768906
print(part2(parsers.string(loader.get())))  # 6363913128533
