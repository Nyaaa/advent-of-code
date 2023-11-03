import hashlib
from itertools import count

from tools import loader, parsers


def start(key: str, part2: bool) -> int:
    """
    >>> start('abcdef', part2=False)
    609043
    >>> start('pqrstuv', part2=False)
    1048970
    """
    for i in count(1):
        h = hashlib.md5(f'{key}{i}'.encode()).hexdigest()
        if h.startswith('00000' if not part2 else '000000'):
            return i
    raise ValueError('Solution not found.')


print(start(parsers.string(loader.get()), part2=False))  # 346386
print(start(parsers.string(loader.get()), part2=True))  # 9958218
