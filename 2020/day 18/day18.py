import re
from math import prod

from tools import parsers, loader


def do_the_math(expr: str) -> int:
    """No precedence, evaluate left to right.
    >>> print(do_the_math('1 + 2 * 3 + 4 * 5 + 6'))
    71"""
    expr = expr.replace(' ', '')
    while _sum := re.search(r'(\d+)([+*])(\d+)', expr):
        a = int(_sum.group(1))
        b = int(_sum.group(3))
        op = _sum.group(2)
        result = a + b if op == '+' else a * b
        expr = f'{expr[:_sum.start()]}{result}{expr[_sum.end():]}'
    return int(expr)


def do_advanced_math(expr: str) -> int:
    """Addition takes precedence over multiplication.
    >>> print(do_advanced_math('1 + 2 * 3 + 4 * 5 + 6'))
    231"""
    expr = expr.replace(' ', '')
    while _sum := re.search(r'(\d+)\+(\d+)', expr):
        result = int(_sum.group(1)) + int(_sum.group(2))
        expr = f'{expr[:_sum.start()]}{result}{expr[_sum.end():]}'
    return prod([int(i) for i in re.findall(r'\d+', expr)])


def do_brackets(expr: str, advanced: bool = False) -> int:
    """
    >>> print(do_brackets('2 * 3 + (4 * 5)'))
    26

    >>> print(do_brackets('5 + (8 * 3 + 9 + 3 * 4 * 3)'))
    437

    >>> print(do_brackets('5 * 9 * (7 * 3 * 3 + 9 * 3 + (8 + 6 * 4))'))
    12240

    >>> print(do_brackets('((2 + 4 * 9) * (6 + 9 * 8 + 6) + 6) + 2 + 4 * 2'))
    13632

    >>> print(do_brackets('2 * 3 + (4 * 5)', advanced=True))
    46

    >>> print(do_brackets('5 + (8 * 3 + 9 + 3 * 4 * 3)', advanced=True))
    1445

    >>> print(do_brackets('5 * 9 * (7 * 3 * 3 + 9 * 3 + (8 + 6 * 4))', advanced=True))
    669060

    >>> print(do_brackets('((2 + 4 * 9) * (6 + 9 * 8 + 6) + 6) + 2 + 4 * 2', advanced=True))
    23340
    """
    while True:
        if i := re.search(r'\([^()]+\)', expr):  # [^()]+ matches all chars except ( and )
            inner = re.sub(r'[()]', '', i[0])
            intermediate = do_the_math(inner) if not advanced else do_advanced_math(inner)
            expr = f'{expr[:i.start()]}{intermediate}{expr[i.end():]}'
        else:
            return do_the_math(expr) if not advanced else do_advanced_math(expr)


def calculate(data: list, part2: bool = False) -> int:
    return sum([do_brackets(line, advanced=part2) for line in data])


print(calculate(parsers.lines(loader.get())))  # 202553439706
print(calculate(parsers.lines(loader.get()), part2=True))  # 88534268715686
