import re
from collections import Counter
from string import ascii_lowercase

from tools import loader, parsers


def part_1(data: list[str]) -> int:
    result = 0
    for line in data:
        line = line.replace('-', '')
        text, checksum = re.findall(r'[a-z]+', line)
        common = sorted(Counter(text).most_common(), key=lambda x: (-x[1], x[0]))
        if checksum == ''.join(i[0] for i in common[:5]):
            result += int(re.findall(r'\d+', line)[0])
    return result


def part_2(data: list[str]) -> int:
    for line in data:
        line = line.replace('-', '')
        text = re.findall(r'([a-z|]+)\d', line)[0]
        _id = int(re.findall(r'\d+', line)[0])
        result = ''
        for i in text:
            letter = ascii_lowercase.index(i)
            result += ascii_lowercase[(letter + _id) % 26]
        if 'north' in result:
            return _id


print(part_1(parsers.lines(loader.get())))  # 409147
print(part_2(parsers.lines(loader.get())))  # 991
