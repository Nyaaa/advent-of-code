from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

from tools import loader, parsers


class CardTypes(Enum):
    high_card = 0
    one_pair = 1
    two_pair = 2
    three_kind = 3
    full_house = 4
    four_kind = 5
    five_kind = 6

    def __lt__(self, other: CardTypes) -> bool:
        return self.value < other.value


@dataclass
class Hand:
    cards: list[int]
    bid: int
    hand_type: CardTypes = field(init=False)

    def __post_init__(self) -> None:
        counts = Counter(self.cards).most_common()
        if 1 in self.cards:
            jokers = self.cards.count(1)
            if jokers != 5:
                counts.remove((1, jokers))
                counts[0] = (counts[0][0], counts[0][1] + jokers)

        if counts[0][1] == 5:
            self.hand_type = CardTypes.five_kind
        elif counts[0][1] == 4:
            self.hand_type = CardTypes.four_kind
        elif counts[0][1] == 3:
            self.hand_type = CardTypes.full_house if counts[1][1] == 2 else CardTypes.three_kind
        elif counts[0][1] == 2:
            self.hand_type = CardTypes.two_pair if counts[1][1] == 2 else CardTypes.one_pair
        else:
            self.hand_type = CardTypes.high_card

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
