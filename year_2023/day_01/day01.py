import regex as re

from tools import loader, parsers


def part_1(data: list[str]) -> int:
    """
    >>> print(part_1(parsers.lines('test.txt')))
    142"""
    result = 0
    for line in data:
        nums = re.findall(r'\d', line)
        result += int(nums[0] + nums[-1])
    return result


def part_2(data: list[str]) -> int:
    """
    >>> print(part_2(parsers.lines('test2.txt')))
    281"""
    words = {'one': '1', 'two': '2', 'three': '3',
             'four': '4', 'five': '5', 'six': '6',
             'seven': '7', 'eight': '8', 'nine': '9'}
    re_words = re.compile('|'.join(words.keys()) + r'|\d')
    result = 0
    for line in data:
        matches = re.findall(re_words, line, overlapped=True)
        nums = [i if i.isdigit() else words[i] for i in matches]
        result += int(nums[0] + nums[-1])
    return result


print(part_1(parsers.lines(loader.get())))  # 55002
print(part_2(parsers.lines(loader.get())))  # 55093
