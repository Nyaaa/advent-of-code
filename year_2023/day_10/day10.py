from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path

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

    def __init__(self, data: list[str]) -> None:
        self.pipes = {}
        self.arr = np.genfromtxt(data, delimiter=1, dtype=str)

        for (i, j), tile in np.ndenumerate(self.arr):
            location = complex(i, j)
            if tile in '-|F7JLS':
                self.pipes[location] = tile
            if tile == 'S':
                self.cart = Cart(location, location)

        self.cart.direction = next(i for i in (1, -1, 1j, -1j)
                                   if self.cart.location + i in self.pipes)
        self.track = self.get_pipe_circle(self.cart)

    def get_pipe_circle(self, cart: Cart) -> list[complex]:
        track = []
        while True:
            track.append(cart.location)
            cart.location += cart.direction
            cart.rotate(self.pipes.get(cart.location))
            if cart.location == cart.start:
                break
        return track

    def part_1(self) -> int:
        """
        >>> print(Pipe(parsers.lines('test.txt')).part_1())
        8"""
        return len(self.track) // 2

    def part_2(self) -> int:
        """
        >>> print(Pipe(parsers.lines('test2.txt')).part_2())
        4
        >>> print(Pipe(parsers.lines('test3.txt')).part_2())
        8"""
        path = Path(np.array([(int(i.real), int(i.imag)) for i in self.track]))
        return sum(
            complex(*point) not in self.track and path.contains_point(point)
            for point, _ in np.ndenumerate(self.arr)
        )


print(Pipe(parsers.lines(loader.get())).part_1())  # 7086
print(Pipe(parsers.lines(loader.get())).part_2())  # 317
