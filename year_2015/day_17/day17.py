from itertools import combinations

from tools import loader, parsers


def containers(data: list[str], target: int) -> tuple[int, int]:
    """
    >>> print(containers(['20', '15', '10', '5', '5'], 25))
    (4, 3)"""
    options = list(map(int, data))
    combos = {}
    for i in range(1, len(options) + 1):
        combos[i] = sum(sum(j) == target for j in (combinations(options, i)))
    return sum(combos.values()), next(a for a in combos.values() if a)


print(containers(parsers.lines(loader.get()), 150))  # 654, 57
