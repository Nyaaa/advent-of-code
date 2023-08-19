from tools import parsers, loader
from queue import LifoQueue


def alu(data: list[str], part2: bool):
    stack = LifoQueue()
    output = {}
    for i in range(14):
        a = int(data[i * 18 + 5][6:])
        if a > 0:
            stack.put((i, int(data[i * 18 + 15][6:])))
        else:
            x, y = stack.get()
            y += a
            output[i] = min(9, 9 + y) if not part2 else max(1, 1 + y)
            output[x] = min(9, 9 - y) if not part2 else max(1, 1 - y)
    return ''.join(str(v) for _, v in sorted(output.items()))


print(alu(parsers.lines(loader.get()), False))  # 79997391969649
print(alu(parsers.lines(loader.get()), True))  # 16931171414113
