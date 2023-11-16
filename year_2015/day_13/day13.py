import re
from collections import defaultdict
from itertools import permutations

from tools import loader, parsers


def optimize_seating(data: list[str], part2: bool) -> int:
    """
    >>> print(optimize_seating(parsers.lines('test.txt'), part2=False))
    330"""
    guests = defaultdict(dict)
    for line in data:
        s = r'(\w+) would (gain|lose) (\d+) happiness units by sitting next to (\w+).'
        info = re.findall(s, line)[0]
        value = '-' if info[1] == 'lose' else '+'
        guests[info[0]][info[3]] = int(value + info[2])
    if part2:
        guests['self'] = {k: 0 for k in guests}
    best_happiness = 0
    for combo in permutations(guests.keys(), len(guests)):
        happiness = 0
        for i, guest in enumerate(combo):
            neighbour = combo[(i + 1) % len(combo)]
            happiness += guests[guest].get(neighbour, 0)
            happiness += guests[neighbour].get(guest, 0)
        best_happiness = max(best_happiness, happiness)
    return best_happiness


print(optimize_seating(parsers.lines(loader.get()), part2=False))  # 733
print(optimize_seating(parsers.lines(loader.get()), part2=True))  # 725
