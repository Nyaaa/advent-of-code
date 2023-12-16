from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from tools import loader, parsers


@dataclass
class Beam:
    boundary: tuple[int, int]
    location: complex = 0j
    direction: complex = 1j
    path: set[complex] = field(default_factory=set)
    infinity = 0

    def rotate(self, tile: str) -> None | Beam:
        new_beam = None
        match tile:
            case '\\':
                self.direction *= -1j if self.direction.real == 0 else 1j
            case '/':
                self.direction *= 1j if self.direction.real == 0 else -1j
            case  '|' if self.direction.real == 0:
                new_beam = Beam(self.boundary, self.location, self.direction * -1j)
                self.direction *= 1j
            case '-' if self.direction.imag == 0:
                new_beam = Beam(self.boundary, self.location, self.direction * -1j)
                self.direction *= 1j
        return new_beam

    def move(self) -> bool:
        self.path.add(self.location)
        self.location += self.direction
        row_valid = 0 <= self.location.real < self.boundary[0]
        col_valid = 0 <= self.location.imag < self.boundary[1]
        if self.location in self.path:
            self.infinity += 1
        if not row_valid or not col_valid or self.infinity > 30:
            return False
        return True


class Contraption:
    def __init__(self, data: list[str]) -> None:
        self.mirrors = {}
        self.boundary = (len(data), len(data[0]))
        for i, row in enumerate(data):
            for j, tile in enumerate(row):
                if tile in r'/\|-':
                    self.mirrors[complex(i, j)] = tile

    def part_1(self, start: Beam = None) -> int:
        """
        >>> print(Contraption(parsers.lines('test.txt')).part_1())
        46"""
        if not start:
            start = Beam(self.boundary)
        beams = deque([start])
        energized = set()
        visited_splits = set()
        while beams:
            beam = beams.popleft()
            alive = True
            while alive:
                energized.add(beam.location)
                tile = self.mirrors.get(beam.location)
                new_beam = beam.rotate(tile)
                if new_beam and new_beam.location not in visited_splits:
                    visited_splits.add(new_beam.location)
                    if new_beam.move():
                        beams.append(new_beam)
                alive = beam.move()
        return len(energized)

    def part_2(self) -> int:
        """
        >>> print(Contraption(parsers.lines('test.txt')).part_2())
        51"""
        best_result = 0
        starts = []
        for i in range(self.boundary[1]):
            starts.append(Beam(self.boundary, complex(0, i), 1))
            starts.append(Beam(self.boundary, complex(self.boundary[0], i), -1))
        for i in range(self.boundary[0]):
            starts.append(Beam(self.boundary, complex(i, 0), 1j))
            starts.append(Beam(self.boundary, complex(i, self.boundary[1]), -1j))
        for start in starts:
            best_result = max(best_result, self.part_1(start))
        return best_result


print(Contraption(parsers.lines(loader.get())).part_1())  # 7482
print(Contraption(parsers.lines(loader.get())).part_2())  # 7896
