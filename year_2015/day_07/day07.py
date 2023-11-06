import operator
import re
from collections import deque

from tools import loader, parsers


def circuit(data: list[str], part2: int) -> int:
    signals = {}
    wires = deque(data)
    ops = {'AND': operator.and_,
           'OR': operator.or_,
           'LSHIFT': operator.lshift,
           'RSHIFT': operator.rshift,
           'NOT': operator.invert}

    while wires:
        line = wires.popleft()
        _input, output = line.split(' -> ')
        try:
            vals_int = [int(v) if not v.isalpha() else signals[v]
                        for v in re.findall(r'\d+|[a-z]+', _input)]
        except KeyError:
            wires.append(line)
            continue

        if op := re.findall(r'[A-Z]+', line):
            signals[output] = ops[op[0]](*vals_int)
        else:
            signals[output] = vals_int[0]
            if output == 'b' and part2:
                signals[output] = part2

    return signals['a']


part1 = circuit(parsers.lines(loader.get()), 0)
print(part1)  # 956
print(circuit(parsers.lines(loader.get()), part1))  # 40149
