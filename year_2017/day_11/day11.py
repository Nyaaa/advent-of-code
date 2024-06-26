from tools import loader, parsers

DIRECTIONS = {
    'n': 1j, 's': -1j,
    'ne': 0.5+0.5j, 'nw': -0.5+0.5j,
    'se': 0.5-0.5j, 'sw': -0.5-0.5j,
}


def walk(data: str) -> tuple[int, int]:
    """
    >>> print(walk('ne,ne,s,s')[0])
    2

    >>> print(walk('se,sw,se,sw,sw')[0])
    3"""
    loc = dist = max_dist = 0
    for step in data.split(','):
        loc += DIRECTIONS[step]
        dist = int(abs(loc.real) + abs(loc.imag))
        max_dist = max(dist, max_dist)
    return dist, max_dist


print(walk(parsers.string(loader.get())))  # 824, 1548
