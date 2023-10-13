import operator
from collections import defaultdict

from tools import loader, parsers

OPERATIONS = {
    '<=': operator.le, '>=': operator.ge,
    '==': operator.eq, '!=': operator.ne,
    '<': operator.lt, '>': operator.gt,
    'inc': operator.add, 'dec': operator.sub,
}


def registers(data: list[str]) -> tuple[int, int]:
    """
    >>> print(registers(parsers.lines('test.txt')))
    (1, 10)"""
    register = defaultdict(int)
    highest = 0
    for line in data:
        reg0, op0, val0, _, reg1, op1, val1 = line.split()
        if OPERATIONS[op1](register[reg1], int(val1)):
            register[reg0] = OPERATIONS[op0](register[reg0], int(val0))
            if register[reg0] > highest:
                highest = register[reg0]
    return max(register.values()), highest


print(registers(parsers.lines(loader.get())))  # 3880, 5035
