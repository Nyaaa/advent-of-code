from hashlib import md5
from itertools import count

from tools import loader, parsers


def start(data: str) -> tuple[str, str]:
    """
    >>> print(start('abc'))
    ('18f47a30', '05ace8e3')"""
    part1 = ''
    part2 = {}
    for i in count():
        string = f'{data}{i}'
        _hash = md5(bytes(string, 'utf-8')).hexdigest()
        if _hash.startswith('00000'):
            pos = _hash[5]
            part1 += pos
            try:
                pos = int(pos)
            except ValueError:
                continue
            if 0 <= pos <= 7 and not part2.get(pos):
                part2[pos] = _hash[6]
        if len(part2) == 8:
            break
    return part1[:8], ''.join(v for k, v in sorted(part2.items()))


print(start(parsers.string(loader.get())))  # c6697b55, 8c35d1ab
