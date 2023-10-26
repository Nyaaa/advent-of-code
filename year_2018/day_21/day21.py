import importlib
import re
from inspect import getmembers, isfunction

from tools import loader, parsers


def underflow(data: list[str]) -> tuple[int, int]:
    """Can't be bothered to optimize this. Bruteforce takes a while."""
    ip_register = int(re.findall(r'\d+', data[0])[0])
    instructions = {i: (v.split()[0], (0, *map(int, re.findall(r'\d+', v))))
                    for i, v in enumerate(data[1:])}
    ops = dict(getmembers(
        importlib.import_module('year_2018.day_16.operators'), isfunction))
    register = [0, 0, 0, 0, 0, 0]

    part1 = 0
    seen = set()
    part2 = 0
    while True:
        ip = register[ip_register]
        op, instruction = instructions[ip]
        register[ip_register] = ip
        register = ops[op](instruction, register)
        ip = register[ip_register] + 1
        if op == 'eqrr':
            value = register[instruction[1]]
            if not part1:
                part1 = value
            if value in seen:
                break
            seen.add(value)
            print(len(seen))
            part2 = value
        if not 0 <= ip < len(instructions):
            break
        register[ip_register] = ip
    return part1, part2


print(underflow(parsers.lines(loader.get())))  # 13522479, 14626276
