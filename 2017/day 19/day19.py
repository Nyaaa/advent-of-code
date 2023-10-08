from tools import loader, parsers


def main(data: list[str]) -> tuple[str, int]:
    """
    >>> print(main(parsers.lines('test.txt', strip=False)))
    ('ABCDEF', 38)"""
    grid = {}
    for i, row in enumerate(data):
        for j, cell in enumerate(row):
            if cell not in {' ', '\n'}:
                grid[complex(i, j)] = cell
    direction = 1-0j
    location = min(grid, key=lambda x: x.real)
    letters = ''
    length = 0
    while value := grid.get(location):
        length += 1
        if value.isalpha():
            letters += value
        elif value == '+':
            direction *= 1j if location + direction * 1j in grid else -1j
        location += direction
    return letters, length


print(main(parsers.lines(loader.get(), strip=False)))  # ('GPALMJSOY', 16204)
