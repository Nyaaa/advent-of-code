import re
from dataclasses import dataclass
from tools import parsers, loader
from itertools import cycle, islice


TEST = """Player 1 starting position: 4
Player 2 starting position: 8
"""


@dataclass
class Player:
    id: int
    place: int
    score: int = 0


def deterministic_dice():
    die = cycle(range(1, 101))
    yield from iter(lambda: list(islice(die, 3)), [])


class Board:
    def __init__(self, data):
        self.players_list = []
        for line in data:
            pl = re.findall(r'\d+', line)
            self.players_list.append(Player(id=int(pl[0]), place=int(pl[1])))
        self.players = cycle(self.players_list)

    def part_1(self):
        """
        >>> print(Board(parsers.inline_test(TEST)).part_1())
        739785"""
        counter = 0
        die = deterministic_dice()

        while True:
            counter += 3
            player = next(self.players)
            roll = sum(next(die))
            place = (roll + player.place) % 10
            if place == 0:
                place = 10
            player.place = place
            player.score += place

            if player.score >= 1000:
                break

        return min([p.score for p in self.players_list]) * counter


print(Board(parsers.inline_test(TEST)).part_1())
print(Board(parsers.lines(loader.get())).part_1())  # 920079
