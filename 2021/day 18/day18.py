from typing import Iterator, Generator
from tools import parsers, loader
import re


def get_objects(string: str, pairs: bool) -> Generator[Iterator[re.Match], None, None]:
    if pairs:
        pattern = re.compile(r'(\[\d+,\d+])')
    else:
        pattern = re.compile(r'\d+')
    yield re.finditer(pattern, string)


def explode(num: str) -> str:
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
        digits = list(*get_objects(side_string, pairs=False))
        try:
            to_replace = digits[side_slice]
        except IndexError:
            return side_string
        else:
            index = to_replace.start()
            new_value = int(value.group()) + int(to_replace.group())
            return f'{side_string[:index]}{new_value}{side_string[index + 1:]}'

    for match in next(get_objects(num, pairs=True)):
        start_pos = match.start()
        depth = num.count('[', 0, start_pos) - num.count(']', 0, start_pos)
        if depth == 4:
            digit_left, digit_right = next(get_objects(match.group(), pairs=False))
            index_left, index_right = match.span()
            left = num[:index_left]
            right = num[index_right:]
            left = add_numbers(digit_left, left, -1)
            right = add_numbers(digit_right, right, 0)
            result = f'{left}0{right}'
            return result
    return num


def do_math(data: list):
    for i in data:
        result = explode(i)
    return result


print(do_math(parsers.inline_test('[[[[[9,8],1],2],3],4]')))
print(loader.get())

