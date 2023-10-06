import re

from tools import loader, parsers


def part_1(data: str) -> int:
    """
    >>> print(part_1('{{<ab>},{<ab>},{<ab>},{<ab>}}'))
    9

    >>> print(part_1('{{<!!>},{<!!>},{<!!>},{<!!>}}'))
    9

    >>> print(part_1('{{{},{},{{}}}}'))
    16"""
    data = re.sub(r'!.', '', data)
    data = re.sub(r'<[^>]*>', '', data)
    out = value = 0
    for i in data:
        if i == '{':
            value += 1
        elif i == '}':
            out += value
            value -= 1
    return out


def part_2(data: str) -> int:
    """
    >>> print(part_2('<{o"i!a,<{i<a>'))
    10"""
    data = re.sub(r'!.', '', data)
    return sum(len(i.group(0)) - 2 for i in re.finditer(r'<[^>]*>', data))


print(part_1(parsers.string(loader.get())))  # 12505
print(part_2(parsers.string(loader.get())))  # 6671
