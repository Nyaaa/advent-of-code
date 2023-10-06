from tools import loader, parsers

TEST = """F10
N3
F7
R90
F11
"""


class Navigation:
    def __init__(self, data: list[str]) -> None:
        self.direction = 1j
        self.position = 0j
        self.waypoint = -1+10j
        self.instructions = [(i[0], int(i[1:])) for i in data]

    @staticmethod
    def rotate(point: complex, heading: str, value: int) -> complex:
        for _ in range(value // 90):
            point *= -1j if heading == 'R' else 1j
        return point

    def get_move(self, heading: str, distance: int) -> complex:
        move = 0j
        match heading:
            case 'F': move = self.direction * distance
            case 'N': move = complex(-distance, 0)
            case 'S': move = complex(distance, 0)
            case 'E': move = complex(0, distance)
            case 'W': move = complex(0, -distance)
            case _: self.direction = self.rotate(self.direction, heading, distance)
        return move

    def part_1(self) -> int:
        """test part 1:
        >>> print(Navigation(parsers.inline_test(TEST)).part_1())
        25"""
        for heading, val in self.instructions:
            self.position += self.get_move(heading, val)
        return int(abs(self.position.real) + abs(self.position.imag))

    def part_2(self) -> int:
        """test part 2:
        >>> print(Navigation(parsers.inline_test(TEST)).part_2())
        286"""
        for heading, val in self.instructions:
            match heading:
                case 'F': self.position += self.waypoint * val
                case 'R' | 'L': self.waypoint = self.rotate(self.waypoint, heading, val)
                case _: self.waypoint += self.get_move(heading, val)
        return int(abs(self.position.real) + abs(self.position.imag))


print(Navigation(parsers.lines(loader.get())).part_1())  # 858
print(Navigation(parsers.lines(loader.get())).part_2())  # 39140
