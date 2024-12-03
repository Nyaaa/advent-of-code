import re
from math import prod

from tools import loader, parsers

TEST = 'xmul(2,4)%&mul[3,7]!@^do_not_mul(5,5)+mul(32,64]then(mul(11,8)mul(8,5))'
TEST2 = "xmul(2,4)&mul[3,7]!^don't()_mul(5,5)+mul(32,64](mul(11,8)undo()?mul(8,5))"


def part1(data: list[str]) -> int:
    """
    >>> print(part1(parsers.inline_test(TEST)))
    161"""
    return sum(prod(map(int, j)) for i in data for j in re.findall(r'mul\((\d+),(\d+)\)', i))


def part2(data: list[str]) -> int:
    """
    >>> print(part2(parsers.inline_test(TEST2)))
    48"""
    result = 0
    op_enabled = True
    for op_string in data:
        operation = re.findall(r"(mul|don't|do)(?:\((\d+,\d+)\)|\(\))", op_string)
        for op, val in operation:
            match op:
                case 'do': op_enabled = True
                case "don't": op_enabled = False
                case 'mul' if op_enabled: result += prod(map(int, val.split(',')))
    return result


print(part1(parsers.lines(loader.get())))  # 184122457
print(part2(parsers.lines(loader.get())))  # 107862689
