from collections import Counter, deque

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

    part1 = 0
    for winning, ticket in tickets.values():
        matches = winning.intersection(ticket)
        value = 0 if not matches else 1
        for _ in range(len(matches) - 1):
            value *= 2
        part1 += value

    queue = deque([*tickets.items()])
    all_tickets = []
    while queue:
        index, (winning, ticket) = queue.popleft()
        all_tickets.append(index)
        matches = len(winning.intersection(ticket))
        for i in range(1, matches + 1):
            next_ticket = index + i
            queue.append((next_ticket, tickets[next_ticket]))
    part2 = Counter(all_tickets).total()

    return part1, part2


print(lottery(parsers.lines(loader.get())))  # 32001, 5037841

