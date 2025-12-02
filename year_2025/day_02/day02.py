import re

from tools import loader, parsers


def find_repetitions(data: str, part2: bool) -> int:
    """
    >>> print(find_repetitions(parsers.string('test.txt'), False))
    1227775554
    >>> print(find_repetitions(parsers.string('test.txt'), True))
    4174379265"""
    result = 0
    pattern = r'^(\d+)\1+$' if part2 else r'^(\d+)\1$'
    regex = re.compile(pattern)
    for line in data.split(','):
        left, right = line.split('-')
        for i in range(int(left), int(right) + 1):
            if re.search(regex, str(i)):
                result += i
    return result


print(find_repetitions(parsers.string(loader.get()), False))  # 8576933996
print(find_repetitions(parsers.string(loader.get()), True))  # 25663320831
