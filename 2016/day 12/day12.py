from tools import parsers, loader


def monorail(data: list[str], value: int = 0) -> int:
    """
    >>> print(monorail(parsers.lines('test.txt')))
    42"""
    commands = [i.split() for i in data]
    memory = {'a': 0, 'b': 0, 'c': value, 'd': 0}
    step = 0
    while step < len(commands):
        match commands[step]:
            case 'cpy', x, y:
                memory[y] = memory[x] if x.islower() else int(x)
            case 'inc', x:
                memory[x] += 1
            case 'dec', x:
                memory[x] -= 1
            case 'jnz', x, y:
                x = memory[x] if x.islower() else int(x)
                if x != 0:
                    step += int(y)
                    continue
        step += 1
    return memory['a']


print(monorail(parsers.lines(loader.get())))  # 318007
print(monorail(parsers.lines(loader.get()), 1))  # 9227661
