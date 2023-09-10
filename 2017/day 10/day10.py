import numpy as np
from tools import parsers, loader
from collections import deque
from more_itertools import chunked


def get_hash(size: int, data: list[int], times: int) -> deque[int]:
    rope = deque(range(size))
    skip = rotated = 0
    for _ in range(times):
        for length in data:
            selected = [rope.popleft() for _ in range(length)]
            rope.extendleft(selected)
            r = length + skip
            rope.rotate(-r)
            rotated += r
            skip += 1
    rope.rotate(rotated)
    return rope


def part_1(data: str, size: int) -> int:
    """
    >>> print(part_1('3,4,1,5', 5))
    12"""
    data = [int(i) for i in data.split(',')]
    result = get_hash(size, data, 1)
    return result.popleft() * result.popleft()


def part_2(data: str) -> str:
    """
    >>> print(part_2('1,2,3'))
    3efbe78a8d82f29979031a4aa0b16a9d"""
    data = [ord(i) for i in data] + [17, 31, 73, 47, 23]
    rope = get_hash(256, data, 64)
    res = [np.bitwise_xor.reduce(i) for i in chunked(rope, 16)]
    return ''.join(f'{i:02x}' for i in res)


if __name__ == '__main__':
    print(part_1(parsers.string(loader.get()), 256))  # 5577
    print(part_2(parsers.string(loader.get())))  # 44f4befb0f303c0bafd085f97741d51d
