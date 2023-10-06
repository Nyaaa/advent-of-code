import re

from tools import loader, parsers


def part_1(data: list) -> int:
    """
    >>> print(part_1(parsers.blocks('test.txt')))
    2"""
    passports = [' '.join(i) for i in data]
    out = 0
    valid_full = ['byr', 'iyr', 'eyr', 'hgt', 'hcl', 'ecl', 'pid']
    for passport in passports:
        if all(i in passport for i in valid_full):
            out += 1
    return out


def part_2(data: list) -> int:
    """
    >>> print(part_2(parsers.blocks('test.txt')))
    2"""
    passports = [' '.join(i) for i in data]
    out = 0
    patterns = re.compile(r"(byr:19[2-9]\d|byr:200[0-2])"
                          r"|(iyr:201\d|iyr:2020)"
                          r"|(eyr:202\d|eyr:2030)"
                          r"|(hgt:1[5-8]\dcm|hgt:19[0-3]cm|hgt:59in|hgt:6\din|hgt:7[0-6]in)"
                          r"|(hcl:#[0-9a-f]{6})"
                          r"|(ecl:(amb|blu|brn|gry|grn|hzl|oth))"
                          r"|(pid:\d{9}\b)")
    for passport in passports:
        result = list(re.finditer(patterns, passport))
        out += 1 if len(result) == 7 else 0
    return out


print(part_1(parsers.blocks(loader.get())))  # 206
print(part_2(parsers.blocks(loader.get())))  # 123
