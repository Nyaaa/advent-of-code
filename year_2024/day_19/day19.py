from functools import cache

from tools import loader, parsers


@cache
def count_arrangements(design: str, towels: frozenset[str]) -> int:
    result = 1 if design in towels else 0
    result += sum(count_arrangements(design=design[len(towel):], towels=towels)
                  for towel in towels if design.startswith(towel))
    return result


def arrange_towels(data: list[list[str]], part2: bool) -> int:
    """
    >>> print(arrange_towels(parsers.blocks('test.txt'), part2=False))
    6
    >>> print(arrange_towels(parsers.blocks('test.txt'), part2=True))
    16"""
    towels_, designs = data
    towels = frozenset(towels_[0].split(', '))
    result = 0
    for design in designs:
        variants = count_arrangements(design=design, towels=towels)
        result += variants if part2 else bool(variants)
    return result


print(arrange_towels(parsers.blocks(loader.get()), part2=False))  # 293
print(arrange_towels(parsers.blocks(loader.get()), part2=True))  # 623924810770264
