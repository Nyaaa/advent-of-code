from tools import parsers, loader
import re

TEST = """1-3 a: abcde
1-3 b: cdefg
2-9 c: ccccccccc
"""


def password_validator(data: list, part2: bool) -> int:
    """test part 1:
    >>> print(password_validator(parsers.inline_test(TEST),False))
    2

    test part 2:
    >>> print(password_validator(parsers.inline_test(TEST),True))
    1"""
    valid = 0
    for line in data:
        char_min, char_max, char, string = re.findall(r'(\d+)-(\d+) ([a-z]): (\w+)', line)[0]
        char_min, char_max = int(char_min), int(char_max)
        if not part2:
            valid += 1 if char_min <= string.count(char) <= char_max else 0
        else:
            valid += 1 if (string[char_min - 1] == char) != (string[char_max - 1] == char) else 0
    return valid


print(password_validator(parsers.lines(loader.get()), False))  # 378
print(password_validator(parsers.lines(loader.get()), True))  # 280
