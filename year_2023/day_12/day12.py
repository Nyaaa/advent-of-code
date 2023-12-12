from functools import cache

from tools import loader, parsers


@cache
def count_variants(string: str, check: tuple[int]) -> int:
    curr_group_length, *next_check = check
    variants = 0
    for index in range(len(string) - sum(next_check) - len(next_check) - curr_group_length + 1):
        left = f'{"." * index}{"#" * curr_group_length}.'
        right = string[len(left):]
        if all(j in (i, '?') for i, j in zip(left, string)):
            if next_check:
                variants += count_variants(right, tuple(next_check))
            elif '#' not in right:
                variants += 1
    return variants


def springs(data: list[str], part2: bool) -> int:
    """
    >>> print(springs(parsers.lines('test.txt'), part2=False))
    21
    >>> print(springs(parsers.lines('test.txt'), part2=True))
    525152"""
    result = 0
    for line in data:
        damaged, check = line.split()
        check = tuple(int(i) for i in check.split(','))
        if part2:
            damaged = '?'.join([damaged] * 5)
            check = (check * 5)
        result += count_variants(damaged, check)
    return result


print(springs(parsers.lines(loader.get()), part2=False))  # 7084
print(springs(parsers.lines(loader.get()), part2=True))  # 8414003326821
