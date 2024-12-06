from tools import loader, parsers


def simulate_movement(grid: dict[complex, str], position: complex) -> set[complex]:
    max_steps = 1000
    direction = -1+0j
    path = {position}
    repeated_steps = 0
    while repeated_steps < max_steps:
        next_step = position + direction
        if next_step in path:
            repeated_steps += 1
        next_space = grid.get(next_step)
        if next_space == '.':
            position = next_step
            path.add(position)
        elif next_space == '#':
            direction *= -1j
        else:
            return path
    raise ValueError


def patrol(data: list[str], part2: bool) -> int:
    """
    >>> print(patrol(parsers.lines('test.txt'), part2=False))
    41
    >>> print(patrol(parsers.lines('test.txt'), part2=True))
    6"""
    grid = {}
    position = 0j
    for row, line in enumerate(data):
        for col, char in enumerate(line):
            pos = complex(row, col)
            grid[pos] = char
            if char == '^':
                position = pos
                grid[pos] = '.'
    path = simulate_movement(grid, position)
    blocks = []

    if part2:
        path.remove(position)
        for pos in path:
            new_grid = grid.copy()
            new_grid[pos] = '#'
            try:
                simulate_movement(new_grid, position)
            except ValueError:
                blocks.append(pos)

    return len(path) if not part2 else len(blocks)


print(patrol(parsers.lines(loader.get()), part2=False))  # 5162
print(patrol(parsers.lines(loader.get()), part2=True))  # 1909
