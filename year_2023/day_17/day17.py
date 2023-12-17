from __future__ import annotations

from collections.abc import Generator
from heapq import heappop, heappush
from typing import NamedTuple

from tools import loader, parsers


class State(NamedTuple):
    location: complex
    direction: complex
    heat_loss: int = 0
    distance: int = 0

    def __lt__(self, other: State) -> bool:
        return self.heat_loss < other.heat_loss

    def __eq__(self, other: State) -> bool:
        return ((self.location, self.direction, self.distance) ==
                (other.location, other.direction, other.distance))

    def __hash__(self) -> int:
        return hash((self.location, self.direction, self.distance))


class Factory:
    def __init__(self, data: list[str]) -> None:
        self.arr = {}
        for i, row in enumerate(data):
            for j, tile in enumerate(row):
                self.arr[complex(i, j)] = int(tile)
        self.destination = complex(len(data) - 1, len(data[0]) - 1)
        self.seen = set()

    def move(self, state: State) -> Generator[State]:
        directions = [1j, -1j]
        if state.distance < 3:
            directions.append(1)
        for direction in directions:
            _dir = state.direction * direction
            loc = state.location + _dir
            if loc not in self.arr:
                continue
            s = State(location=loc,
                      direction=_dir,
                      heat_loss=state.heat_loss + self.arr[loc],
                      distance=1 if _dir != state.direction else state.distance + 1)
            if s not in self.seen:
                self.seen.add(s)
                yield s

    def part_1(self) -> int:
        """
        >>> print(Factory(parsers.lines('test.txt')).part_1())
        102"""
        queue = [State(0j, 1j), State(0j, 1)]
        while queue:
            state = heappop(queue)
            if state.location == self.destination:
                return state.heat_loss
            for s in self.move(state):
                heappush(queue, s)
        raise ValueError('Solution not found')


print(Factory(parsers.lines(loader.get())).part_1())  # 942