from tools import parsers, loader
from string import ascii_lowercase


def part_1(data: list[str]):
    """
    >>> print(part_1(['dabAcCaCBAcCcaDA']))
    10"""
    out = []
    for letter in data[0]:
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


def part_2(data: list[str]):
    """
    >>> print(part_2(['dabAcCaCBAcCcaDA']))
    4"""
    shortest = float('inf')
    for i in ascii_lowercase:
        new = data[0].replace(i, '').replace(i.upper(), '')
        result = part_1([new])
        if result < shortest:
            shortest = result
    return shortest


print(part_1(parsers.lines(loader.get())))  # 10584
print(part_2(parsers.lines(loader.get())))  # 6968
