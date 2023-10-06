import re
from collections import defaultdict

from tools import loader, parsers


def start(data: list[list[str]]) -> int:
    """
    >>> print(start(parsers.blocks('test.txt')))
    3"""
    state = re.findall(r' ([A-Z])\.', data[0][0])[0]
    steps = int(re.findall(r'\d+', data[0][1])[0])
    tape = defaultdict(int)
    instructions = {}
    for i in data[1:]:
        values = [line.split()[-1][:-1] for line in i]
        instructions[(values[0], 0)] = values[2:5]
        instructions[(values[0], 1)] = values[6:]
    position = 0
    for _ in range(steps):
        value = tape[position]
        operation = instructions.get((state, value))
        tape[position] = int(operation[0])
        position += 1 if operation[1] == 'right' else -1
        state = operation[2]
    return sum(tape.values())


print(start(parsers.blocks(loader.get())))  # 2870
