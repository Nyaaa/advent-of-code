import math
from collections import deque

import regex as re

from tools import loader, parsers


def lottery(data: list[str]) -> tuple[int, int]:
    """
    >>> print(lottery(parsers.lines('test.txt')))
    (13, 30)"""
    tickets = {}
    for i, line in enumerate(data, start=1):
        winning, ticket = line.split(':')[1].split('|')
        winning = set(map(int, re.findall(r'\d+', winning)))
        ticket = set(map(int, re.findall(r'\d+', ticket)))
        tickets[i] = (winning, ticket)

    part1 = part2 = 0
    for winning, ticket in tickets.values():
        if matches := winning.intersection(ticket):
            part1 += int(math.pow(2, len(matches) - 1))

    queue = deque([*tickets])
    seen = {}
    while queue:
        index = queue.popleft()
        part2 += 1
        if index not in seen:
            winning, ticket = tickets[index]
            matches = len(winning.intersection(ticket))
            new_tickets = [index + i + 1 for i in range(matches)]
            seen[index] = new_tickets
        queue.extend(seen[index])

    return part1, part2


print(lottery(parsers.lines(loader.get())))  # 32001, 5037841
