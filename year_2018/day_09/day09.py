import re
from collections import defaultdict, deque
from itertools import cycle

from tools import loader, parsers


def marbles(data: str, part2: bool) -> int:
    """
    >>> print(marbles('10 players; last marble is worth 1618 points', False))
    8317"""

    players, last_marble = (int(i) for i in re.findall(r'\d+', data))
    circle = deque([0])
    scores = defaultdict(int)
    players = cycle(range(players))
    if part2:
        last_marble *= 100

    for marble in range(1, last_marble + 1):
        player = next(players)
        if marble % 23 == 0:
            circle.rotate(7)
            scores[player] += marble + circle.pop()
            circle.rotate(-1)
        else:
            circle.rotate(-1)
            circle.append(marble)
    return max(scores.values())


print(marbles(parsers.string(loader.get()), False))  # 367634
print(marbles(parsers.string(loader.get()), True))  # 3020072891
