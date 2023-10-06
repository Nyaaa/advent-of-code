from __future__ import annotations

import re
from collections import Counter, deque
from dataclasses import dataclass, field
from functools import cache
from itertools import permutations, product
from typing import NamedTuple, NoReturn

from tools import loader, parsers, timer

DATA = parsers.blocks(loader.get())
TEST = parsers.blocks('test.txt')
# Precalculate all possible permutations of x, y, z
ROTATIONS = []
for r in permutations(['x', 'y', 'z']):
    for s in product(['-', ''], repeat=3):
        ROTATIONS.append([''.join(i) for i in zip(s, r)])


class Point(NamedTuple):
    x: int
    y: int
    z: int

    def __sub__(self, other: Point) -> Point:
        return Point(self.x - other.x, self.y - other.y, self.z - other.z)

    def __add__(self, other: Point) -> Point:
        return Point(self.x + other.x, self.y + other.y, self.z + other.z)

    def manhattan(self, other: Point) -> int:
        return abs(self.x - other.x) + abs(self.y - other.y) + abs(self.z - other.z)


@dataclass
class Scanner:
    ident: int
    scans: frozenset[Point]
    rotations: list[frozenset[Point]] = field(init=False)
    location: Point | None = None

    def __repr__(self) -> str:
        return f'Scanner {self.ident}'

    def __post_init__(self) -> None:
        """Calculate scan variants"""
        self.rotations = []
        for rotation in ROTATIONS:
            new_rotation = []
            for point in self.scans:
                axes = {'x': point.x, 'y': point.y, 'z': point.z,
                        '-x': -point.x, '-y': -point.y, '-z': -point.z}
                new_rotation.append(Point(*(axes[i] for i in rotation)))
            self.rotations.append(frozenset(new_rotation))

    def locate(self, known_scanner: Scanner) -> NoReturn:
        for rotation in self.rotations:
            location, count = get_counts(rotation, known_scanner.scans)
            if count == 12:
                self.location = location
                self.scans = frozenset(self.location + scan for scan in rotation)
                break
        else:
            raise ValueError('No solution found')


@cache
def get_counts(rotation: frozenset[Point], scans: frozenset[Point]) -> tuple[Point, int]:
    counts = [point_2 - point_1 for point_2 in scans for point_1 in rotation]
    return Counter(counts).most_common(1)[0]


def locate_scanners(scanners: list[Scanner]) -> list[Scanner]:
    lost = deque(scanners)
    first_scanner = lost.popleft()
    first_scanner.location = Point(0, 0, 0)
    found = [first_scanner]

    while lost:
        unknown_scanner = lost.popleft()
        for known_scanner in found:
            try:
                unknown_scanner.locate(known_scanner)
            except ValueError:
                continue
            found.append(unknown_scanner)
            break
        else:
            lost.append(unknown_scanner)
    return found


def parse(data: list[list[str]]) -> list[Scanner]:
    scanners = []
    for block in data:
        scanner_id = 0
        scans = []
        for line in block:
            nums = re.findall(r'\d+|-\d+', line)
            if len(nums) == 1:
                scanner_id = int(nums[0])
            else:
                scans.append(Point(*(int(i) for i in nums)))
        scanners.append(Scanner(ident=scanner_id, scans=frozenset(scans)))
    return scanners


def start(data: list[Scanner]) -> tuple[int, int]:
    """
    >>> print(start(parse(TEST)))
    (79, 3621)"""
    located = locate_scanners(data)
    # flatten nested list comprehension
    flatten = [element for sublist in (i.scans for i in located) for element in sublist]
    part_1 = len(set(flatten))

    part_2 = 0
    for i, j in permutations(located, 2):
        distance = i.location.manhattan(j.location)
        if distance > part_2:
            part_2 = distance
    return part_1, part_2


with timer.context():
    print(start(parse(DATA)))  # 353, 10832
