import operator
from collections import defaultdict
from itertools import combinations

from tools import loader, parsers


def manhattan_distance(a: complex, b: complex) -> int:
    return int(abs(a.real - b.real) + abs(a.imag - b.imag))


def get_nodes_part1(a: complex, b: complex, d: complex, rows: range, cols: range) -> set[complex]:
    result = set()
    nodes = [a - d, a + d, b - d, b + d]
    for n in nodes:
        a1 = manhattan_distance(a, n)
        b1 = manhattan_distance(b, n)
        if a1 == 0 or b1 == 0:
            continue
        if (b1 // a1 == 2 or a1 // b1 == 2) and (n.real in rows and n.imag in cols):
            result.add(n)
    return result


def get_nodes_part2(a: complex, b: complex, d: complex, rows: range, cols: range) -> set[complex]:
    result = set()
    for p in (a, b):
        for op in (operator.add, operator.sub):
            current = p
            while True:
                new = op(current, d)
                if new.real in rows and new.imag in cols:
                    result.add(new)
                    current = new
                else:
                    break
    return result


def get_antinodes(points: set[complex], rows: range, cols: range, part2: bool) -> set[complex]:
    result = set()
    for a, b in combinations(points, 2):
        d = complex(a.real - b.real, a.imag - b.imag)
        get_nodes = get_nodes_part1 if not part2 else get_nodes_part2
        result.update(get_nodes(a, b, d, rows, cols))
    return result


def count_antinodes(data: list[str], part2: bool) -> int:
    """
    >>> print(count_antinodes(parsers.lines('test.txt'), part2=False))
    14
    >>> print(count_antinodes(parsers.lines('test2.txt'), part2=True))
    9
    >>> print(count_antinodes(parsers.lines('test.txt'), part2=True))
    34"""
    grid = defaultdict(set)
    rows = range(len(data))
    cols = range(len(data[0]))
    for row, line in enumerate(data):
        for col, char in enumerate(line):
            if char == '.':
                continue
            pos = complex(row, col)
            grid[char].add(pos)
    result = set()

    for v in grid.values():
        nodes = get_antinodes(v, rows, cols, part2)
        result.update(nodes)
    return len(result)


print(count_antinodes(parsers.lines(loader.get()), part2=False))  # 398
print(count_antinodes(parsers.lines(loader.get()), part2=True))  # 1333
