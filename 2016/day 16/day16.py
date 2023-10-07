from more_itertools import grouper

from tools import loader, parsers


def gen_data(a: str) -> str:
    """
    >>> print(gen_data('111100001010'))
    1111000010100101011110000"""
    b = ''.join('1' if i == '0' else '0' for i in reversed(a))
    return f'{a}0{b}'


def gen_checksum(data: str) -> str:
    """
    >>> print(gen_checksum('110010110100'))
    100"""
    data = ''.join('1' if a == b else '0' for a, b in grouper(data, 2))
    return gen_checksum(data) if len(data) % 2 == 0 else data


def fill(data: str, size: int) -> str:
    """
    >>> print(fill('10000', 20))
    01100"""
    while len(data) <= size:
        data = gen_data(data)
    return gen_checksum(data[:size])


print(fill(parsers.string(loader.get()), 272))  # 10100101010101101
print(fill(parsers.string(loader.get()), 35651584))  # 01100001101101001
