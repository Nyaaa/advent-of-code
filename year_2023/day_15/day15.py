import contextlib
import re
from collections import defaultdict
from functools import reduce

from tools import loader, parsers

TEST = 'rn=1,cm-,qp=3,cm=2,qp-,pc=4,ot=9,ab=5,pc-,pc=6,ot=7'


def get_hash(step: str) -> int:
    """
    >>> print(get_hash('HASH'))
    52"""
    return reduce(lambda x, y: (x + ord(y)) * 17 % 256, step, 0)


def hasher(data: str) -> tuple[int, int]:
    """
    >>> print(hasher(TEST))
    (1320, 145)"""
    steps = data.split(',')
    part1 = part2 = 0
    boxes = defaultdict(dict)
    for step in steps:
        part1 += get_hash(step)
        label, op, val = re.findall(r'(\w+)([=-])(\d+)?', step)[0]
        box = get_hash(label)
        if op == '-':
            with contextlib.suppress(KeyError):
                boxes[box].pop(label)
        else:
            boxes[box][label] = int(val)
    for box, lenses in boxes.items():
        for i, value in enumerate(lenses.values(), start=1):
            part2 += (1 + box) * i * value
    return part1, part2


print(hasher(parsers.string(loader.get())))  # 512797, 262454
