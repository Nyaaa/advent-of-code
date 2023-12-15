from functools import reduce

from tools import loader, parsers

TEST = 'rn=1,cm-,qp=3,cm=2,qp-,pc=4,ot=9,ab=5,pc-,pc=6,ot=7'


def get_hash(step: str) -> int:
    """
    >>> print(get_hash('HASH'))
    52"""
    return reduce(lambda x, y: (x + ord(y)) * 17 % 256, step, 0)


def hasher(data: str) -> int:
    """
    >>> print(hasher(TEST))
    1320"""
    steps = data.split(',')
    return sum(get_hash(step) for step in steps)


print(hasher(parsers.string(loader.get())))  # 512797
