import re

from tools import loader, parsers


def decompress(line: str, part2: bool) -> int:
    """
    >>> print(decompress('X(8x2)(3x3)ABCY', False))
    18

    >>> print(decompress('(25x3)(3x3)ABC(2x3)XY(5x2)PQRSTX(18x9)(3x2)TWO(5x7)SEVEN', True))
    445"""
    line_len = 0
    while line:
        marker = re.search(r'\((\d+)x(\d+)\)', line)
        if not marker:
            line_len += len(line)
            break
        chars, times = map(int, marker.groups())
        line_len += len(line[:marker.start()])
        if not part2:
            repeat_len = len(line[marker.end():][:chars])
        else:
            repeat_len = decompress(line[marker.end():][:chars], True)
        line_len += (repeat_len * times)
        line = line[marker.end() + chars:]
    return line_len


print(decompress(parsers.string(loader.get()), False))  # 112830
print(decompress(parsers.string(loader.get()), True))  # 10931789799
