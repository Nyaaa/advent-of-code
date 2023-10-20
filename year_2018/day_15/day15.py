from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from tools import loader, parsers


@dataclass
class Unit:
    position: complex
    team: int  # 0 = E, 1 = G
    hp: int = 200
    atk: int = 3
    alive: bool = True

    def distance(self, other: Unit) -> float:
        return (abs(self.position.real - other.position.real) +
                abs(self.position.imag - other.position.imag))

    def in_range(self, other: Unit) -> bool:
        return self.distance(other) <= 1


class Battle:
    def __init__(self, data: list[str]) -> None:
        self.grid = set()
        self.units = []
        for row, line in enumerate(data):
            for col, char in enumerate(line):
                if char not in (' ', '#'):
                    self.grid.add(complex(row, col))
                if char.isalpha():
                    self.units.append(Unit(complex(row, col), 0 if char == 'E' else 1))

    def find_path(
            self, unit: Unit, enemies: list[Unit]
    ) -> tuple[int | None, list[complex] | None]:
        next_step = None
        paths = deque([(unit.position, 0, next_step)])
        t = {target.position + j for j in (1, 1j, -1j, -1) for target in enemies}
        targets = t.intersection(self.grid)
        unit_positions = {i.position for i in self.units if i != unit and i.alive}
        found = []
        min_dist = float('inf')
        seen = set()

        while paths:
            current, distance, next_step = paths.popleft()
            if distance == 1:
                next_step = current
            if current in seen or distance > min_dist:
                continue
            seen.add(current)
            if current in targets:
                found.append((current, distance, next_step))
                min_dist = distance

            for n in (-1, -1j, 1j, 1):
                adj = current + n
                if adj in self.grid and adj not in unit_positions:
                    paths.append((adj, distance + 1, next_step))

        if found:
            return min(found, key=lambda x: (x[1], x[0].real, x[0].imag))[1:]
        return None, None

    def cycle(self, part2: bool = False) -> int:
        """
        >>> print(Battle(parsers.lines('test.txt')).cycle())
        27730

        >>> print(Battle(parsers.lines('test2.txt')).cycle())
        36334

        >>> print(Battle(parsers.lines('test3.txt')).cycle())
        27755

        >>> print(Battle(parsers.lines('test4.txt')).cycle())
        28944

        >>> print(Battle(parsers.lines('test5.txt')).cycle())
        18740"""
        rounds = 0
        while True:
            units = sorted([i for i in self.units if i.alive],
                           key=lambda u: (u.position.real, u.position.imag))
            for unit in units:
                if not unit.alive:
                    continue
                enemies = [i for i in units if i.team != unit.team and i.alive]
                if not enemies:
                    return rounds * sum([i.hp for i in units if i.alive])
                closest_distance, closest_path = self.find_path(unit, enemies)
                if closest_path and closest_distance > 0:
                    unit.position = closest_path
                    closest_distance -= 1
                if closest_distance == 0:
                    targets = [i for i in units if
                               i.team != unit.team and i.distance(unit) == 1 and i.alive]
                    target = min(targets, key=lambda u: (u.hp, u.position.real, u.position.imag))
                    if target:
                        target.hp -= unit.atk
                        if target.hp <= 0:
                            target.alive = False
                            if part2 and target.team == 0:
                                return 0
            rounds += 1


def part_2(data: list[str]) -> int:
    power = 4
    result = 0
    while result == 0:
        battle = Battle(data)
        for elf in (i for i in battle.units if i.team == 0):
            elf.atk = power
        result = battle.cycle(True)
        power += 1
    return result


print(Battle(parsers.lines(loader.get())).cycle())  # 189000
print(part_2(parsers.lines(loader.get())))  # 38512
