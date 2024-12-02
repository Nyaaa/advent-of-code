from itertools import combinations

from tools import loader, parsers


def check_reactor(data: list[str], part2: bool) -> int:
    """
    >>> print(check_reactor(parsers.lines('test.txt'), part2=False))
    2
    >>> print(check_reactor(parsers.lines('test.txt'), part2=True))
    4"""
    data = [tuple(map(int, line.split())) for line in data]
    result = 0
    for line in data:
        variants = [line]
        if part2:
            variants.extend(combinations(line, len(line) - 1))
        total_safety = 0
        for variant in variants:
            srt = tuple(sorted(variant))
            srt2 = tuple(reversed(srt))
            if variant not in {srt, srt2}:
                continue
            for i, j in enumerate(variant[1:-1], start=1):
                left_diff = abs(variant[i - 1] - j)
                right_diff = abs(j - variant[i + 1])
                if not (1 <= left_diff <= 3 and 1 <= right_diff <= 3):
                    break
            else:
                total_safety += 1
        if total_safety > 0:
            result += 1
    return result


print(check_reactor(parsers.lines(loader.get()), part2=False))  # 486
print(check_reactor(parsers.lines(loader.get()), part2=True))  # 540
