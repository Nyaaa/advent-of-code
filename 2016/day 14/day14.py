import re
from collections.abc import Generator
from tools import parsers, loader
import hashlib
from itertools import count


def part1_hasher(salt: str) -> Generator[tuple[int, str]]:
    c = count()
    while True:
        index = next(c)
        string = f'{salt}{index}'
        h = hashlib.md5(string.encode())
        yield index, h.hexdigest()


def part2_hasher(salt: str):
    _hasher = part1_hasher(salt)
    while True:
        i, h = next(_hasher)
        for _ in range(2016):
            h = hashlib.md5(h.encode()).hexdigest()
        yield i, h


def find_keys(hasher):
    keys = []
    candidates = []
    while len(keys) < 64:
        index, _hash = next(hasher)

        candidates = [i for i in candidates if i[1] + 1000 >= index]
        for cand, i in candidates:
            quintuplet = re.search(fr'({cand})\1\1\1\1', _hash)
            if quintuplet:
                keys.append(i)

        triplet = re.search(r'(.)\1\1', _hash)
        if triplet:
            candidates.append((triplet.group(1), index))
    return sorted(keys)[63]


def start(salt: str, part2: bool) -> int:
    """
    >>> print(start('abc', False))
    22728

    >>> print(start('abc', True))
    22551"""
    if not part2:
        hasher = part1_hasher(salt)
    else:
        hasher = part2_hasher(salt)
    return find_keys(hasher)


print(start(parsers.string(loader.get()), False))  # 16106
print(start(parsers.string(loader.get()), True))  # 22423
