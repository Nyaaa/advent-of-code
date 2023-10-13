from string import ascii_lowercase

from tools import loader, parsers


def part_1(data: str) -> int:
    """
    >>> print(part_1('dabAcCaCBAcCcaDA'))
    10"""
    out = []
    for letter in data:
        try:
            previous = out[-1]
        except IndexError:
            out.append(letter)
            continue
        if previous.swapcase() == letter:
            out.pop()
        else:
            out.append(letter)
    return len(out)


def part_2(data: str) -> int:
    """
    >>> print(part_2('dabAcCaCBAcCcaDA'))
    4"""
    shortest = float('inf')
    for i in ascii_lowercase:
        new = data.replace(i, '').replace(i.upper(), '')
        result = part_1(new)
        if result < shortest:
            shortest = result
    return shortest


print(part_1(parsers.string(loader.get())))  # 10584
print(part_2(parsers.string(loader.get())))  # 6968
