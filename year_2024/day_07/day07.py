import operator
import re
from collections.abc import Callable
from itertools import product

from more_itertools import roundrobin

from tools import loader, parsers


def concatenate_int(a: int, b: int) -> int:
    return int(str(a) + str(b))


def evaluate(expr: list[int | Callable], expect: int) -> int:
    while len(expr) > 1:
        a, op, b = expr[:3]
        new_val = op(a, b)
        if new_val > expect:
            return -1
        expr = [new_val] + expr[3:]
    return expr[0]


def part1(data: list[str], part2: bool) -> int:
    """
    >>> print(part1(parsers.lines('test.txt'), part2=False))
    3749
    >>> print(part1(parsers.lines('test.txt'), part2=True))
    11387"""
    ops = [operator.add, operator.mul]
    if part2:
        ops.append(concatenate_int)
    result = 0
    for line in data:
        expect, *nums = list(map(int, re.findall(r'\d+', line)))
        for i in product(ops, repeat=len(nums)-1):
            expr = list(roundrobin(nums, i))
            if evaluate(expr, expect) == expect:
                result += expect
                break
    return result


print(part1(parsers.lines(loader.get()), part2=False))  # 3312271365652
print(part1(parsers.lines(loader.get()), part2=True))  # 509463489296712
