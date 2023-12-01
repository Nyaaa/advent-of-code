import regex as re

from tools import loader, parsers


def calibration(data: list[str], part2: bool = False) -> int:
    """
    >>> print(calibration(parsers.lines('test.txt'), part2=False))
    142
    >>> print(calibration(parsers.lines('test2.txt'), part2=True))
    281"""
    words = {'one': '1', 'two': '2', 'three': '3',
             'four': '4', 'five': '5', 'six': '6',
             'seven': '7', 'eight': '8', 'nine': '9'}
    re_words = re.compile('|'.join(words.keys()) + r'|\d')
    result = 0
    for line in data:
        matches = re.findall(re_words if part2 else r'\d', line, overlapped=True)
        nums = [i if i.isdigit() else words[i] for i in matches]
        result += int(nums[0] + nums[-1])
    return result


print(calibration(parsers.lines(loader.get()), part2=False))  # 55002
print(calibration(parsers.lines(loader.get()), part2=True))  # 55093
