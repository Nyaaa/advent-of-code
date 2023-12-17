from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass, field
from itertools import cycle

from tools import loader, parsers


@dataclass
class Cart:
    ident: int
    location: complex
    direction: complex
    turn: cycle[complex] = field(default_factory=lambda: cycle([1j, 1, -1j]))

    def __eq__(self, other: Cart) -> bool:
        return self.location == other.location

    def __lt__(self, other: Cart) -> bool:
        return ((self.location.imag, self.location.real)
                < (other.location.imag, other.location.real))

    def __str__(self) -> str:
        return f'{self.ident} L: {self.location} D: {self.direction}'

    def loc_str(self) -> str:
        return f'{int(self.location.imag)},{int(self.location.real)}'

    def __hash__(self) -> int:
        return hash(self.location)

    def rotate(self, tile: str) -> None:
        match tile:
            case '+':
                self.direction *= next(self.turn)
            case '\\':
                self.direction *= -1j if self.direction.real == 0 else 1j
            case '/':
                self.direction *= 1j if self.direction.real == 0 else -1j


class Tracks:
    def __init__(self, data: list[str]) -> None:
        self.DIRECTIONS = {'^': -1, 'v': 1, '<': -1j, '>': 1j}
        self.map = {}
        self.carts = []
        _id = 1

        for i, row in enumerate(data):
            for j, tile in enumerate(row):
                location = complex(i, j)
                if tile in r'/\+':
                    self.map[location] = tile
                elif tile in self.DIRECTIONS:
                    self.carts.append(Cart(_id, location, self.DIRECTIONS[tile]))
                    _id += 1

    def simulate(self) -> Generator[Cart]:
        while len(self.carts) > 1:
            for cart in sorted(self.carts):
                cart.location += cart.direction
                cart.rotate(self.map.get(cart.location))
                if self.carts.count(cart) > 1:
                    self.carts = [i for i in self.carts if i.location != cart.location]
                    yield cart

    def part_1(self) -> str:
        """
        >>> print(Tracks(parsers.lines('test.txt')).part_1())
        7,3"""
        return next(self.simulate()).loc_str()

    def part_2(self) -> str:
        """
        >>> print(Tracks(parsers.lines('test2.txt', False)).part_2())
        6,4"""
        list(self.simulate())
        return self.carts[0].loc_str()


print(Tracks(parsers.lines(loader.get(), False)).part_1())  # 45,34
print(Tracks(parsers.lines(loader.get(), False)).part_2())  # 91,25
