from __future__ import annotations

from dataclasses import dataclass

from tools import loader, parsers


@dataclass
class Cart:
    location: complex
    start: complex
    direction: complex = 0j

    def rotate(self, tile: str) -> None:
        match tile:
            case 'L' | '7':
                self.direction *= -1j if self.direction.real == 0 else 1j
            case 'J' | 'F':
                self.direction *= 1j if self.direction.real == 0 else -1j


class Pipe:
    """Code borrowed from 2018 day 13"""
    adjacent = (1, -1, 1j, -1j)

    def __init__(self, data: list[str]) -> None:
        self.map = {}

        for i, row in enumerate(data):
            for j, tile in enumerate(row):
                location = complex(i, j)
                if tile in r'[-|F7JLS]':
                    self.map[location] = tile
                if tile == 'S':
                    self.cart = Cart(location, location)
        self.cart.direction = next(i for i in Pipe.adjacent if self.cart.location + i in self.map)

    def simulate(self, cart: Cart) -> list[complex]:
        track = []
        while True:
            track.append(cart.location)
            cart.location += cart.direction
            cart.rotate(self.map.get(cart.location))
            if cart.location == cart.start:
                break
        return track

    def part_1(self) -> int:
        """
        >>> print(Pipe(parsers.lines('test.txt')).part_1())
        8"""
        track = self.simulate(self.cart)
        return len(track) // 2


print(Pipe(parsers.lines(loader.get())).part_1())  # 7086
