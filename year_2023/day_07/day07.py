from __future__ import annotations

from collections import Counter, deque
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


@dataclass
class Hand:
    cards: list[int]
    bid: int
    rank: int = 0
    hand_type: CardTypes = field(init=False)

    def __post_init__(self) -> None:
        counts = list(Counter(self.cards).values())
        if 5 in counts:
            self.hand_type = CardTypes.five_kind
        elif 4 in counts:
            self.hand_type = CardTypes.four_kind
        elif 3 in counts and 2 in counts:
            self.hand_type = CardTypes.full_house
        elif 3 in counts:
            self.hand_type = CardTypes.three_kind
        elif counts.count(2) == 2:
            self.hand_type = CardTypes.two_pair
        elif 2 in counts:
            self.hand_type = CardTypes.one_pair
        else:
            self.hand_type = CardTypes.high_card

    def __lt__(self, other: Hand) -> bool:
        for i, val in enumerate(self.cards):
            if val != other.cards[i]:
                return val < other.cards[i]
        raise ValueError('Hands are identical')


def camel_cards(data: list[str]) -> int:
    """
    >>> print(camel_cards(parsers.lines('test.txt')))
    6440"""
    card_values = {v: k for k, v in enumerate('23456789TJQKA', start=2)}
    hands = [Hand(cards=[card_values[i] for i in k], bid=int(v))
             for k, v in (x.split() for x in data)]
    hand_rank = 1
    for card_type in CardTypes:
        queue = deque([h for h in hands if h.hand_type == card_type])
        while queue:
            left = queue.popleft()
            if not queue:
                left.rank = hand_rank
                hand_rank += 1
                break
            is_smallest = all(left < i for i in queue)
            if not is_smallest:
                queue.append(left)
            else:
                left.rank = hand_rank
                hand_rank += 1
            pass
    return sum(i.rank * i.bid for i in hands)


# print(camel_cards(parsers.lines('test.txt')))
print(camel_cards(parsers.lines(loader.get())))  # 249483956
