from tools import parsers, loader
from copy import deepcopy


def load(commands: list):
    accumulator = 0
    index = 0
    while index < len(commands):
        op, val, seen = commands[index]
        commands[index][2] = True
        match op, seen:
            case _, True:
                return accumulator, False
            case 'nop', _:
                index += 1
            case 'acc', _:
                accumulator += val
                index += 1
            case 'jmp', _:
                index += val
    return accumulator, True


def part_1(data: list) -> int:
    """ test part 1:
    >>> print(part_1(parsers.lines('test.txt')))
    5"""
    return load([[k, int(v), False] for k, v in (line.split() for line in data)])[0]


def part_2(data: list) -> int:
    """ test part 2:
    >>> print(part_2(parsers.lines('test.txt')))
    8"""
    commands = [[k, int(v), False] for k, v in (line.split() for line in data)]
    seen = []
    for i, val in enumerate(commands):
        new_cmd = deepcopy(commands)
        if val not in seen:
            if val[0] == 'nop':
                new_cmd[i] = ['jmp', val[1], val[2]]
            elif val[0] == 'jmp':
                new_cmd[i] = ['nop', val[1], val[2]]
            result = load(new_cmd)
            seen.append(val)
            if result[1]:
                return result[0]
    raise ValueError('Solution not found')


print(part_1(parsers.lines(loader.get())))  # 1475
print(part_2(parsers.lines(loader.get())))  # 1270
