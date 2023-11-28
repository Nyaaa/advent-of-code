import re
from dataclasses import dataclass
from itertools import combinations
from math import ceil

from tools import loader, parsers

WEAPONS = [(8, 4, 0), (10, 5, 0), (25, 6, 0), (40, 7, 0), (74, 8, 0)]
ARMOUR = [(13, 0, 1), (31, 0, 2), (53, 0, 3), (75, 0, 4), (102, 0, 5), (0, 0, 0)]
RINGS = [(25, 1, 0), (50, 2, 0), (100, 3, 0),
         (20, 0, 1), (40, 0, 2), (80, 0, 3), (0, 0, 0), (0, 0, 0)]


@dataclass
class Unit:
    hp: int = 100
    dmg: int = 0
    ac: int = 0


def fight(player: Unit, boss: Unit) -> Unit:
    player_outcome = ceil(player.hp / max(1, boss.dmg - player.ac))
    boss_outcome = ceil(boss.hp / max(1, player.dmg - boss.ac))
    return player if player_outcome >= boss_outcome else boss


def battle(data: str) -> tuple[int, int]:
    items = []
    for wep in WEAPONS:
        for arm in ARMOUR:
            for ring1, ring2 in combinations(RINGS, 2):
                cost = wep[0] + arm[0] + ring1[0] + ring2[0]
                damage = wep[1] + arm[1] + ring1[1] + ring2[1]
                armor = wep[2] + arm[2] + ring1[2] + ring2[2]
                items.append((cost, damage, armor))
    part1 = part2 = 0

    for cost, dmg, ac in sorted(items):
        player = Unit(hp=100, dmg=dmg, ac=ac)
        boss = Unit(*map(int, re.findall(r'\d+', data)))
        winner = fight(player, boss)
        if winner == player and not part1:
            part1 = cost
        if winner == boss and cost > part2:
            part2 = cost
    return part1, part2


print(battle(parsers.string(loader.get())))  # 78, 148
