from __future__ import annotations
from collections import defaultdict, deque
from typing import NamedTuple
from tools import parsers, loader
import re
from itertools import count, combinations


class State(NamedTuple):
    elevator: int
    floors: frozenset[complex]
    steps: int = 0

    def __eq__(self, other: State) -> bool:
        return self.floors == other.floors and self.elevator == other.elevator

    def __hash__(self) -> int:
        return hash(self.floors) + self.elevator

    def is_valid(self) -> bool:
        """
        >>> print(State(0, frozenset({1j, -1j, -2j, 3j})).is_valid())
        False

        >>> print(State(0, frozenset({-1j, 2+2j})).is_valid())
        True

        >>> print(State(0, frozenset({(3+1j), (2-1j), (3-2j), 2j})).is_valid())
        False

        >>> print(State(0, frozenset({-1j, 1j, 2j})).is_valid())
        False"""
        for i in range(4):
            gens = [j for j in self.floors if i == j.real and j.imag < 0]
            chips = [j for j in self.floors if i == j.real and j.imag > 0 and complex(i, -j.imag) not in self.floors]
            if gens and chips:
                return False
        return True


class Facility:
    ADDITION = 'elerium generator, elerium-compatible microchip, dilithium generator, dilithium-compatible microchip'

    def __init__(self, data: list[str], part2: bool = False) -> None:
        floors = set()
        nums = count(1)
        elements = defaultdict(lambda: next(nums))
        for i, floor in enumerate(data):
            if i == 0 and part2:
                floor += Facility.ADDITION
            for item in re.finditer(r'(\w+)(?:-\w+)? (microchip|generator)', floor):
                element = elements[item.group(1)]
                x = element if item.group(2) == 'microchip' else -element
                floors.add(complex(i, x))
        self.initial_state = State(0, frozenset(floors))
        self.end_state = State(3, frozenset(complex(3, i.imag) for i in self.initial_state.floors))

    def start(self) -> int:
        """
        >>> print(Facility(parsers.lines('test.txt')).start())
        11"""
        seen = set()
        queue = deque([self.initial_state])
        while queue:
            state = queue.popleft()
            if state == self.end_state:
                return state.steps
            if state in seen:
                continue
            seen.add(state)
            items = [i for i in state.floors if i.real == state.elevator]
            moves = [c for i in (1, 2) for c in combinations(items, i)]
            if state.elevator == 0 or all(i.real >= state.elevator for i in state.floors):
                dirs = [1]
            elif state.elevator == 3:
                dirs = [-1]
            else:
                dirs = [-1, 1]

            for d in dirs:
                for item in moves:
                    _floors = frozenset(i + d if i in item else i for i in state.floors)
                    _state = State(state.elevator + d, _floors, state.steps + 1)
                    if _state.is_valid():
                        queue.append(_state)

        raise ValueError('Path not found')


print(Facility(parsers.lines(loader.get())).start())  # 33
print(Facility(parsers.lines(loader.get()), True).start())  # 57, very slow!
