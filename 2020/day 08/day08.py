from tools import parsers, loader


def part_1(data: list):
    """ test part 1:
    >>> print(part_1(parsers.lines('test.txt')))
    5"""
    commands = [[k, int(v), False] for k, v in (line.split() for line in data)]
    accumulator = 0
    index = 0
    while True:
        op, val, seen = commands[index]
        commands[index][2] = True
        match op, seen:
            case _, True:
                return accumulator
            case 'nop', _:
                index += 1
            case 'acc', _:
                accumulator += val
                index += 1
            case 'jmp', _:
                index += val


print(part_1(parsers.lines(loader.get())))  # 1475
