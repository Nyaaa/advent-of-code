from functools import cache
from itertools import cycle, islice, product
from typing import Iterator

from tools import loader, parsers

TEST = """Player 1 starting position: 4
Player 2 starting position: 8
"""


def deterministic_dice() -> Iterator[int]:
    die = cycle(range(1, 101))
    yield from iter(lambda: sum(islice(die, 3)), [])


@cache
def dirac_dice() -> list[int]:
    rolls = product((1, 2, 3), repeat=3)
    return [sum(roll) for roll in rolls]


def get_players(data: list[str]) -> list[tuple[int, int]]:
    return [(int(line.split(' ')[-1]), 0) for line in data]


def update_player(player: tuple[int, int], roll: int) -> tuple[int, int]:
    position = (player[0] + roll - 1) % 10 + 1
    score = player[1] + position
    return position, score


def part_1(data: list[str]) -> int:
    """
    >>> print(part_1(parsers.inline_test(TEST)))
    739785"""
    counter = 0
    die = deterministic_dice()
    player1, player2 = get_players(data)

    while True:
        if player1[1] >= 1000 or player2[1] >= 1000:
            break
        counter += 3
        player1, player2 = player2, update_player(player1, next(die))

    return min(player1[1], player2[1]) * counter


@cache
def dirac(player1: tuple[int, int], player2: tuple[int, int]) -> tuple[int, int]:
    if player1[1] >= 21:
        return 1, 0
    if player2[1] >= 21:
        return 0, 1
    player1_wins = 0
    player2_wins = 0
    for roll in dirac_dice():
        wins_2, wins_1 = dirac(player2, update_player(player1, roll))
        player1_wins += wins_1
        player2_wins += wins_2
    return player1_wins, player2_wins


def part_2(data: list[str]) -> int:
    """
    >>> print(part_2(parsers.inline_test(TEST)))
    444356092776315"""
    players = get_players(data)
    return max(dirac(*players))


print(part_1(parsers.lines(loader.get())))  # 920079
print(part_2(parsers.lines(loader.get())))  # 56852759190649
