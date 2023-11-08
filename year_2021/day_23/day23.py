from __future__ import annotations

import re
from collections.abc import Generator
from heapq import heappop, heappush

from more_itertools import minmax

from tools import loader, parsers


class State:
    energy = {'A': 1, 'B': 10, 'C': 100, 'D': 1000}
    hallway_positions = [0, 1, 3, 5, 7, 9, 10]
    entrances = {'A': 2, 'B': 4, 'C': 6, 'D': 8}

    def __init__(self, rooms: list[str], energy_spent: int) -> None:
        self.rooms = rooms
        self.energy_spent = energy_spent
        self.target_rooms = {r: range(ord(r) - 54, len(self.rooms), 4) for r in 'ABCD'}
        self.target_indices = {v: key for key, val in self.target_rooms.items() for v in val}

    def __eq__(self, other: State) -> bool:
        return self.rooms == other.rooms

    def __hash__(self) -> int:
        return hash(str(self.rooms))

    def __lt__(self, other: State) -> bool:
        return self.energy_spent < other.energy_spent

    def path_is_valid(self, a: int, b: int) -> bool:
        step = 1 if a < b else -1
        return not any(self.rooms[target] for target in range(a + step, b + step, step))

    def enter_hallway(self, source: int) -> Generator[int]:
        yield from (target for target in self.hallway_positions
                    if not self.rooms[target] and self.path_is_valid(source, target))

    def enter_room(self, source: int, unit: str) -> int | None:
        target = None
        for pos in self.target_rooms[unit]:
            if not self.rooms[pos]:
                target = pos
            elif self.rooms[pos] != unit:
                return None
        if not self.path_is_valid(source, self.entrances[unit]):
            return None
        return target

    def generate_new_moves(self) -> Generator[State]:
        for source in (s for s in self.hallway_positions if self.rooms[s]):
            if target := self.enter_room(source, self.rooms[source]):
                yield self.generate_new_state(source, target)

        for room in 'ABCD':
            source = next((s for s in self.target_rooms[room] if self.rooms[s]), None)
            if source:
                for target in self.enter_hallway(self.entrances[room]):
                    yield self.generate_new_state(source, target)

    def generate_new_state(self, a: int, b: int) -> State:
        _a, _b = minmax(a, b)
        distance = abs(self.entrances[self.target_indices[_b]] - _a) + (_b - 7) // 4
        new_cost = self.energy_spent + distance * self.energy[self.rooms[a]]
        new_rooms = self.rooms.copy()
        new_rooms[a], new_rooms[b] = new_rooms[b], new_rooms[a]
        return State(new_rooms, new_cost)


def simulate(data: str, part2: bool) -> int:
    """
    >>> print(simulate(parsers.string('test.txt'), part2=False))
    12521
    >>> print(simulate(parsers.string('test.txt'), part2=True))
    44169"""
    initial_rooms = re.findall(r'\w', data)
    hallway = [''] * 11
    if part2:
        initial_rooms[-4:-4] = ['D', 'C', 'B', 'A', 'D', 'B', 'A', 'C']
    initial_rooms = hallway + initial_rooms
    start_state = State(initial_rooms, 0)
    end_state = State(hallway + ['A', 'B', 'C', 'D'] * (4 if part2 else 2), 0)

    heap = [start_state]
    seen = {start_state: 0}
    while heap:
        state = heappop(heap)
        if state == end_state:
            return state.energy_spent
        for new_state in state.generate_new_moves():
            if seen.get(new_state, float('inf')) <= new_state.energy_spent:
                continue
            seen[new_state] = new_state.energy_spent
            heappush(heap, new_state)
    raise ValueError('Solution not found.')


print(simulate(parsers.string(loader.get()), part2=False))  # 15237
print(simulate(parsers.string(loader.get()), part2=True))  # 47509
