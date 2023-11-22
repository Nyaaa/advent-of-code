from __future__ import annotations

import heapq
import re
from copy import deepcopy
from dataclasses import dataclass, field

from tools import loader, parsers

BOSS_HP, BOSS_ATTACK = map(int, re.findall(r'\d+', parsers.string(loader.get())))


@dataclass
class State:
    mana_used: int = 0
    boss_hp: int = BOSS_HP
    boss_attack: int = BOSS_ATTACK
    player_hp: int = 50
    player_armour: int = 0
    player_mana: int = 500
    boss_turn: bool = False
    active_spells: list[Spell] = field(default_factory=list)

    def __lt__(self, other: State) -> bool:
        return self.mana_used < other.mana_used


@dataclass
class Spell:
    name: str
    cost: int
    damage: int = 0
    heal: int = 0
    armour: int = 0
    mana: int = 0
    timer: int = 0

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other: Spell) -> bool:
        return self.name == other.name

    def cast(self, state: State) -> bool:
        if (self.timer and self in state.active_spells) or state.player_mana < self.cost:
            return False
        state.player_mana -= self.cost
        state.mana_used += self.cost
        state.player_hp += self.heal
        if self.timer > 0:
            state.active_spells.append(deepcopy(self))
        else:
            self.proc(state)
        return True

    def proc(self, state: State) -> None:
        self.timer -= 1
        state.boss_hp -= self.damage
        state.player_mana += self.mana
        if self.name == 'Shield':
            state.player_armour = self.armour
            if self.timer == 0:
                state.player_armour = 0
        state.active_spells = [i for i in state.active_spells if i.timer > 0]


SPELLS = [
    Spell(name='Magic Missile', cost=53, damage=4),
    Spell(name='Drain', cost=73, damage=2, heal=2),
    Spell(name='Shield', cost=113, armour=7, timer=6),
    Spell(name='Poison', cost=173, damage=3, timer=6),
    Spell(name='Recharge', cost=229, mana=101, timer=5),
]


def battle(part2: bool) -> int:
    queue = [State()]
    while queue:
        state = heapq.heappop(queue)
        if state.player_hp <= 0:
            continue
        if part2 and not state.boss_turn:
            state.player_hp -= 1

        for spell in state.active_spells:
            spell.proc(state)

        if state.boss_hp <= 0:
            return state.mana_used

        if state.boss_turn:
            state.player_hp -= max(1, state.boss_attack - state.player_armour)
            state.boss_turn = False
            heapq.heappush(queue, state)
            continue

        for spell in SPELLS:
            new_state = deepcopy(state)
            new_state.boss_turn = True
            if spell.cast(new_state):
                heapq.heappush(queue, new_state)
    raise ValueError('Solution not found')


print(battle(part2=False))  # 900
print(battle(part2=True))  # 1216
