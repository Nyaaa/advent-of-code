from itertools import permutations

from tools import loader, parsers


def part_1(data: list[str], password: str) -> str:
    """
    >>> print(part_1(parsers.lines('test.txt'),'abcde'))
    decab"""
    pw = list(password)
    for command in data:
        match command.split():
            case 'swap', 'position', x, _, _, y:
                x, y = int(x), int(y)
                pw[x], pw[y] = pw[y], pw[x]
            case 'swap', 'letter', x, _, _, y:
                x, y = pw.index(x), pw.index(y)
                pw[x], pw[y] = pw[y], pw[x]
            case 'rotate', 'left', x, _:
                pw = pw[int(x):] + pw[:int(x)]
            case 'rotate', 'right', x, _:
                pw = pw[-int(x):] + pw[:-int(x)]
            case 'rotate', 'based', *_, y:
                pos = pw.index(y)
                rot = (pos + (2 if pos >= 4 else 1)) % len(pw)
                pw = pw[-rot:] + pw[:-rot]
            case 'move', _, x, _, _, y:
                x, y = int(x), int(y)
                letter = pw.pop(x)
                pw.insert(y, letter)
            case 'reverse', _, x, _, y:
                x, y = int(x), int(y)
                pw[x:y + 1] = reversed(pw[x:y + 1])
    return ''.join(pw)


def part_2(data: list[str], password: str) -> str:
    for p in permutations(password):
        if part_1(data, p) == password:
            return ''.join(p)
    raise ValueError('No solution found.')


print(part_1(parsers.lines(loader.get()), 'abcdefgh'))  # bgfacdeh
print(part_2(parsers.lines(loader.get()), 'fbgdceah'))  # bdgheacf
