import regex as re

from tools import loader, parsers


def part_1(data: list[str]) -> int:
    """
    >>> print(part_1(parsers.lines('test.txt')))
    2"""
    out = 0
    for line in data:
        valid = []
        invalid = []
        for i, word in enumerate(re.findall(r'\w+', line)):
            abba = re.search(r'(.)(?!\1)(.)\2\1', word) is not None
            if i % 2 == 0:
                valid.append(abba)
            else:
                invalid.append(abba)
        if any(valid) and not any(invalid):
            out += 1
    return out


def part_2(data: list[str]) -> int:
    """
    >>> print(part_2(parsers.lines('test.txt')))
    3"""
    def aba(line: str) -> bool:
        inside = set(re.findall(r'\[(\w+)]', line))
        outside = set(re.findall(r'\w+', line)) - inside
        for word in outside:
            for _aba in re.finditer(r'(.)(?!\1)(.)\1', word, overlapped=True):
                _aba = _aba.group()
                bab = f'{_aba[1]}{_aba[0]}{_aba[1]}'
                if any(bab in i for i in inside):
                    return True
        return False
    return sum(aba(line) for line in data)


print(part_1(parsers.lines(loader.get())))  # 110
print(part_2(parsers.lines(loader.get())))  # 242
