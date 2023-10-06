from tools import loader, parsers


def jumps(data: list[str], part2: bool) -> int:
    """
    >>> print(jumps(['0', '3', '0', '1', '-3'], False))
    5

    >>> print(jumps(['0', '3', '0', '1', '-3'], True))
    10"""
    commands = [int(i) for i in data]
    index = 0
    counter = 0
    while index < len(commands):
        pos = commands[index]
        commands[index] += -1 if part2 and pos >= 3 else 1
        index += pos
        counter += 1
    return counter


print(jumps(parsers.lines(loader.get()), False))  # 325922
print(jumps(parsers.lines(loader.get()), True))  # 24490906
