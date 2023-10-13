import re
from collections import defaultdict

from more_itertools import minmax

from tools import loader, parsers


def factory(data: list[str]) -> tuple[str, int]:
    bots = defaultdict(list)
    instructions = {}
    for line in data:
        if line.startswith('value'):
            value, bot = re.findall(r'\d+', line)
            bots['bot ' + bot].append(int(value))
        else:
            instr = re.findall(r'\w+\s\d+', line)
            instructions[instr[0]] = tuple(instr[1:])
    part1 = None

    while True:
        for bot, values in bots.copy().items():
            if len(values) == 2:
                low, high = minmax(values)
                if low == 17 and high == 61:
                    part1 = bot
                low_to, high_to = instructions[bot]
                bots[low_to].append(low)
                bots[high_to].append(high)
                del bots[bot]

        a = bots['output 0']
        b = bots['output 1']
        c = bots['output 2']
        if a and b and c:
            break
    return part1, a[0] * b[0] * c[0]


print(factory(parsers.lines(loader.get())))  # 47, 2666
