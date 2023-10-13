from tools import loader, parsers


def find_seat(line: str) -> int:
    """
    >>> print(find_seat('FBFBBFFRLR'))
    357

    >>> print(find_seat('BFFFBBFRRR'))
    567

    >>> print(find_seat('FFFBBBFRRR'))
    119

    >>> print(find_seat('BBFFBBFRLL'))
    820"""
    line = list(line)
    row = (0, 127)
    col = (0, 7)
    for i in line:
        r_middle = (row[1] - row[0]) // 2
        c_middle = (col[1] - col[0]) // 2
        match i:
            case 'F':
                row = (row[0], row[1] - r_middle - 1)
            case 'B':
                row = (row[0] + r_middle + 1, row[1])
            case 'L':
                col = (col[0], col[1] - c_middle - 1)
            case 'R':
                col = (col[0] + c_middle + 1, col[1])
    return row[0] * 8 + col[0]


def start(data: list[str]) -> tuple[int, int]:
    ids = [find_seat(line) for line in data]
    ids.sort()
    set_actual = set(ids)
    set_complete = set(range(ids[0], ids[-1] + 1))
    return max(ids), *set_complete.difference(set_actual)


print(start(parsers.lines(loader.get())))  # 850, 599
