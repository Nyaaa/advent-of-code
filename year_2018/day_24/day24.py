from __future__ import annotations

import enum
import re
from dataclasses import dataclass
from itertools import count

from tools import loader, parsers


class AttackType(enum.Enum):
    slashing = 0
    bludgeoning = 1
    fire = 2
    cold = 3
    radiation = 4


@dataclass
class UnitGroup:
    group_id: int
    team: int
    units: int
    hp: int
    atk: int
    atk_type: AttackType
    initiative: int
    weakness: frozenset[AttackType]
    immunity: frozenset[AttackType]
    target: UnitGroup = None

    @property
    def effective_power(self) -> int:
        return self.atk * self.units

    def __hash__(self) -> int:
        return hash((self.team, self.group_id))

    def __str__(self) -> str:
        return f'Team {self.team} group {self.group_id}'

    def potential_damage(self, attacker: UnitGroup) -> int:
        damage = attacker.effective_power
        if attacker.atk_type in self.immunity:
            damage = 0
        elif attacker.atk_type in self.weakness:
            damage *= 2
        return damage


class Battle:
    def __init__(self, data: list[list[str]]) -> None:
        self.groups = []
        for block in data:
            _id = count(1)
            team = 0 if block[0] == 'Immune System:' else 1
            for line in block[1:]:
                weak = immune = frozenset()
                units, hp, atk, initiative = map(int, re.findall(r'\d+', line))
                if w := re.findall(r'weak to (.*?)[;)]', line):
                    weak = frozenset(AttackType[i] for i in w[0].split(', '))
                if i := re.findall(r'immune to (.*?)[;)]', line):
                    immune = frozenset(AttackType[i] for i in i[0].split(', '))
                atk_type = AttackType[re.findall(r'does \d+ (\w+)', line)[0]]
                self.groups.append(
                    UnitGroup(next(_id), team, units, hp, atk, atk_type, initiative, weak, immune))

    def part_1(self) -> int:
        """
        >>> print(Battle(parsers.blocks('test.txt')).part_1())
        5216"""
        while True:
            targeted = set()
            self.groups.sort(key=lambda g: (g.effective_power, g.initiative), reverse=True)
            for attacker in self.groups:
                attacker.target = None
                enemies = [(i, i.potential_damage(attacker))
                           for i in self.groups if i.team != attacker.team]
                if not enemies:
                    return sum(i.units for i in self.groups)
                targetable = [i for i in enemies if i[1] > 0 and i[0] not in targeted]
                if not targetable:
                    continue
                target, max_damage = max(
                    targetable, key=lambda g: (g[1], g[0].effective_power, g[0].initiative))
                targeted.add(target)
                attacker.target = target
            self.groups.sort(key=lambda g: g.initiative, reverse=True)
            for attacker in self.groups.copy():
                if not attacker.target or attacker.units <= 0:
                    continue
                killed = attacker.target.potential_damage(attacker) // attacker.target.hp
                attacker.target.units -= killed
                if attacker.target.units <= 0:
                    self.groups.remove(attacker.target)


print(Battle(parsers.blocks(loader.get())).part_1())  # 16090
