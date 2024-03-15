from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from tools import loader, parsers


@dataclass
class Hand:
    cards: list[int]
    bid: int
    hand_type: int = 0

    def __post_init__(self) -> None:
        counts = Counter(self.cards).most_common()
        if 1 in self.cards:
            jokers = self.cards.count(1)
            if jokers != 5:
                counts.remove((1, jokers))
                counts[0] = (counts[0][0], counts[0][1] + jokers)

        match counts[0][1]:
            case 5:
                self.hand_type = 6
            case 4:
                self.hand_type = 5
            case 3:
                self.hand_type = 4 if counts[1][1] == 2 else 3
            case 2:
                self.hand_type = 2 if counts[1][1] == 2 else 1

    def __lt__(self, other: Hand) -> bool:
        if self.hand_type != other.hand_type:
            return self.hand_type < other.hand_type
        for i, val in enumerate(self.cards):
            if val != other.cards[i]:
                return val < other.cards[i]
        raise ValueError('Hands are identical')


def camel_cards(data: list[str], part2: bool) -> int:
    """
    >>> print(camel_cards(parsers.lines('test.txt'), part2=False))
    6440
    >>> print(camel_cards(parsers.lines('test.txt'), part2=True))
    5905"""
    card_values = {v: k for k, v in enumerate('23456789TJQKA', start=2)}
    if part2:
        card_values['J'] = 1
    hands = [Hand(cards=[card_values[i] for i in k], bid=int(v))
             for k, v in (x.split() for x in data)]
    return sum(i * j.bid for i, j in enumerate(sorted(hands), start=1))


print(camel_cards(parsers.lines(loader.get()), part2=False))  # 249483956
print(camel_cards(parsers.lines(loader.get()), part2=True))  # 252137472
