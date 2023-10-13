from collections import deque

from tools import loader, parsers


def part_1(elves: str) -> int:
    """
    >>> print(part_1('5'))
    3"""
    elf = 0
    party = deque(range(1, int(elves) + 1))
    while party:
        party.rotate(-1)
        elf = party.popleft()
    return elf


def part_2(elves: str) -> int:
    """
    >>> print(part_2('5'))
    2"""
    elves = int(elves)
    half_a = deque(range(1, elves // 2 + 1))
    half_b = deque(range(elves // 2 + 1, elves + 1))

    while half_a and half_b:
        half_a.pop() if len(half_a) > len(half_b) else half_b.popleft()
        half_b.append(half_a.popleft())
        half_a.append(half_b.popleft())
    return half_a[0] or half_b[0]


print(part_1(parsers.string(loader.get())))  # 1842613
print(part_2(parsers.string(loader.get())))  # 1424135
