import hashlib
import re
from collections.abc import Generator
from itertools import count

from tools import loader, parsers


def part1_hasher(salt: str) -> Generator[tuple[int, str]]:
    c = count()
    while True:
        index = next(c)
        string = f'{salt}{index}'
        h = hashlib.md5(string.encode())
        yield index, h.hexdigest()


def part2_hasher(salt: str) -> Generator[tuple[int, str]]:
    _hasher = part1_hasher(salt)
    while True:
        i, h = next(_hasher)
        for _ in range(2016):
            h = hashlib.md5(h.encode()).hexdigest()
        yield i, h


def find_keys(salt: str, part2: bool) -> int:
    """
    >>> print(find_keys('abc', False))
    22728

    >>> print(find_keys('abc', True))
    22551"""
    hasher = part2_hasher(salt) if part2 else part1_hasher(salt)
    keys = []
    candidates = []
    while len(keys) < 64:
        index, _hash = next(hasher)

        candidates = [i for i in candidates if i[1] + 1000 >= index]
        for cand, i in candidates:
            if re.search(fr'({cand})\1\1\1\1', _hash):
                keys.append(i)

        if triplet := re.search(r'(.)\1\1', _hash):
            candidates.append((triplet.group(1), index))
    return sorted(keys)[63]


print(find_keys(parsers.string(loader.get()), False))  # 16106
print(find_keys(parsers.string(loader.get()), True))  # 22423
