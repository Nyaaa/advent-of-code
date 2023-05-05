import math
from tools import parsers, loader
import re

PAIRS = re.compile(r'(\[\d+,\d+])')
SINGLES = re.compile(r'\d+')


def get_halves(num: str, delimiter: re.Match) -> tuple[str, str]:
    index_left, index_right = delimiter.span()
    return num[:index_left], num[index_right:]


def explode(num: str) -> str | None:
    """
    >>> print(explode('[[[[[9,8],1],2],3],4]'))
    [[[[0,9],2],3],4]

    >>> print(explode('[7,[6,[5,[4,[3,2]]]]]'))
    [7,[6,[5,[7,0]]]]

    >>> print(explode('[[6,[5,[4,[3,2]]]],1]'))
    [[6,[5,[7,0]]],3]

    >>> print(explode('[[3,[2,[1,[7,3]]]],[6,[5,[4,[3,2]]]]]'))
    [[3,[2,[8,0]]],[9,[5,[4,[3,2]]]]]

    >>> print(explode('[[3,[2,[8,0]]],[9,[5,[4,[3,2]]]]]'))
    [[3,[2,[8,0]]],[9,[5,[7,0]]]]
    """
    def add_numbers(value: re.Match, side_string: str, side_slice: int) -> str:
        digits = list(SINGLES.finditer(side_string))
        if digits:
            to_replace: re.Match = digits[side_slice]
            index = to_replace.span()
            new_value = int(value.group()) + int(to_replace.group())
            return f'{side_string[:index[0]]}{new_value}{side_string[index[1]:]}'
        else:
            return side_string

    for match in PAIRS.finditer(num):
        start_pos = match.start()
        depth = num.count('[', 0, start_pos) - num.count(']', 0, start_pos)
        if depth == 4:
            digit_left, digit_right = SINGLES.finditer(match.group())
            left, right = get_halves(num, match)
            left = add_numbers(digit_left, left, -1)
            right = add_numbers(digit_right, right, 0)
            result = f'{left}0{right}'
            return result
    return None


def split(num: str) -> str | None:
    """
    >>> print(split('10'))
    [5,5]

    >>> print(split('[1,10]'))
    [1,[5,5]]

    >>> print(split('[11,1]'))
    [[5,6],1]"""
    for match in re.finditer(r'\d\d+', num):  # >= 10
        value = match.group()
        left, right = get_halves(num, match)
        num_int = int(value) / 2
        num_a = math.floor(num_int)
        num_b = math.ceil(num_int)
        result = f'{left}[{num_a},{num_b}]{right}'
        return result
    return None


def reduce(num: str):
    """
    >>> print(reduce('[[[[[4,3],4],4],[7,[[8,4],9]]],[1,1]]'))
    [[[[0,7],4],[[7,8],[6,0]]],[8,1]]"""
    while True:
        exploded = explode(num)
        if exploded:
            num = exploded
            continue
        splt = split(num)
        if splt:
            num = splt
        else:
            return num


def snailfish_sum(data: list) -> str:
    """
    >>> print(snailfish_sum(['[1,2]','[[3,4],5]']))
    [[1,2],[[3,4],5]]

    >>> print(snailfish_sum(parsers.lines('test.txt')))
    [[[[8,7],[7,7]],[[8,6],[7,7]]],[[[0,7],[6,6]],[8,7]]]"""
    result = data[0]
    for i in data[1:]:
        result = f'[{result},{i}]'
        result = reduce(result)
    return result



