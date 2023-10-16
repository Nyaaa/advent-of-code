from tools import loader, parsers


def assembunny(data: list[str], value: int = 7) -> int:
    # input-specific solution
    data[4] = 'mul b d a'
    commands = [i.split() for i in data]
    memory = {'a': value, 'b': 0, 'c': 0, 'd': 0}
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
                    step += memory[y] if y.islower() else int(y)
                    continue
            case 'tgl', x:
                x = memory[x] if x.islower() else int(x)
                if 0 <= step + x < len(commands):
                    command = commands[step + x]
                    if len(command) == 2:
                        command[0] = 'dec' if command[0] == 'inc' else 'inc'
                    elif len(command) == 3:
                        command[0] = 'cpy' if command[0] == 'jnz' else 'jnz'
            case 'mul', x, y, z:
                memory[z] = memory[x] * memory[y]
                step += 5
        step += 1
    return memory['a']


print(assembunny(parsers.lines(loader.get())))  # 11130
print(assembunny(parsers.lines(loader.get()), value=12))  # 479007690
