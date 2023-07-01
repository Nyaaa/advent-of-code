import re

from tools import parsers, loader


def do_the_math(expr: str) -> int:
    """
    >>> print(do_the_math('1 + 2 * 3 + 4 * 5 + 6'))
    71"""
    numbers = re.findall(r'\d+', expr)
    operations = re.findall(r' (\D) ', expr)
    result = 0
    step = 0
    for i in operations:
        a = int(numbers[step]) if step == 0 else result
        b = int(numbers[step + 1])
        match i:
            case '+':
                result = (a + b)
            case '*':
                result = (a * b)
        step += 1
    return result


def do_brackets(expr: str) -> int:
    """
    >>> print(do_brackets('2 * 3 + (4 * 5)'))
    26

    >>> print(do_brackets('5 + (8 * 3 + 9 + 3 * 4 * 3)'))
    437

    >>> print(do_brackets('5 * 9 * (7 * 3 * 3 + 9 * 3 + (8 + 6 * 4))'))
    12240

    >>> print(do_brackets('((2 + 4 * 9) * (6 + 9 * 8 + 6) + 6) + 2 + 4 * 2'))
    13632"""
    while True:
        i = re.search(r'\([^()]+\)', expr)
        if i:
            inner = re.sub(r'[()]', '', i[0])
            expr = f'{expr[:i.start()]} {do_the_math(inner)} {expr[i.end():]}'
        else:
            return do_the_math(expr)


def part_1(data: list) -> int:
    result = 0
    for line in data:
        result += do_brackets(line)
    return result


print(part_1(parsers.lines(loader.get())))  # 202553439706

