from collections import deque

from tools import loader, parsers


def is_walkable(location: complex, number: int) -> bool:
    x, y = location.real, location.imag
    val = x * x + 3 * x + 2 * x * y + y + y * y + number
    ones = format(int(val), 'b').count('1')
    return ones % 2 == 0 and x >= 0 and y >= 0


def maze(target: complex, number: int) -> tuple[int, int]:
    """
    >>> print(maze(7+4j, 10))
    (11, 151)"""
    queue = deque([(1+1j, 0)])
    seen = {}
    directions = (1j, -1j, 1, -1)
    while queue:
        location, steps = queue.popleft()
        if location in seen or not is_walkable(location, number):
            continue
        seen[location] = steps
        queue.extend((location + d, steps + 1) for d in directions)
    return seen[target], sum(i <= 50 for i in seen.values())


print(maze(31+39j, int(parsers.string(loader.get()))))  # 90, 135
