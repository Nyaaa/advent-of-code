from collections import deque
from tools import parsers, loader
import hashlib


def get_doors(path: str) -> dict[str, bool]:
    _hash = hashlib.md5(path.encode()).hexdigest()[:4]
    return dict(zip('UDLR', [i in 'bcdef' for i in _hash]))


def find_path(passcode: str) -> tuple[str, int]:
    """
    >>> print(find_path('ulqzkmiv'))
    ('DRURDRUDDLLDLUURRDULRLDUUDDDRR', 830)"""
    location = 0j
    directions = {'U': -1, 'D': 1, 'L': -1j, 'R': 1j}
    path = ''
    found_paths = []
    queue = deque([(location, path)])
    while queue:
        location, path = queue.popleft()
        if location == 3+3j:
            found_paths.append(path)
            continue
        for direction, walkable in get_doors(passcode + path).items():
            new_loc = location + directions[direction]
            if 0 <= new_loc.real <= 3 and 0 <= new_loc.imag <= 3 and walkable:
                queue.append((new_loc, path + direction))
    return found_paths[0], len(found_paths[-1])


print(find_path(parsers.string(loader.get())))  # RRRLDRDUDD, 706
