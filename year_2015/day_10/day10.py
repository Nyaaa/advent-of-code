from itertools import chain, groupby

from tools import loader, parsers


def look_and_say(sequence: str, cycles: int) -> int:
    """
    >>> print(look_and_say('1', 40))
    82350"""
    sequence = list(map(int, sequence))
    for _ in range(cycles):
        sequence = list(chain.from_iterable((len(list(j)), i) for i, j in groupby(sequence)))
    return len(sequence)


print(look_and_say(parsers.string(loader.get()), 40))  # 492982
print(look_and_say(parsers.string(loader.get()), 50))  # 6989950
