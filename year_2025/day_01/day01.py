from tools import loader, parsers

LIMIT = 100


def dial(data: list[str]) -> tuple[int, int]:
    """
    >>> print(dial(parsers.lines('test.txt')))
    (3, 6)"""
    index = 50
    part_1 = part_2 = 0
    for line in data:
        distance = int(line[1:])
        if line[0] == 'L':
            offset = index
            index = (index - distance) % LIMIT
        else:
            offset = (LIMIT - index) % LIMIT
            index = (index + distance) % LIMIT

        if offset == 0:
            part_1 += 1
            offset = LIMIT

        if (diff := distance - offset) >= 0:
            part_2 += diff // LIMIT + 1

    return part_1, part_2


print(dial(parsers.lines(loader.get())))  # (1043, 5963)
