from tools import loader, parsers


def execute(reg_a: int) -> int:
    program = parsers.lines(loader.get())
    register = {'a': reg_a, 'b': 0}
    step = 0
    while 0 <= step < len(program):
        offset = 1
        match program[step].split():
            case 'hlf', x:
                register[x] //= 2
            case 'tpl', x:
                register[x] *= 3
            case 'inc', x:
                register[x] += 1
            case 'jmp', x:
                offset = int(x)
            case 'jie', x, y if register[x[0]] % 2 == 0:
                offset = int(y)
            case 'jio', x, y if register[x[0]] == 1:
                offset = int(y)
        step += offset
    return register['b']


print(execute(reg_a=0))  # 307
print(execute(reg_a=1))  # 160
