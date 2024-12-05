from collections import defaultdict

from tools import loader, parsers


def sort_pages(rules: list[tuple[int, ...]], pages: list[int]) -> list[int]:
    ordr = defaultdict(set)
    for i in pages:
        for left, right in rules:
            if left not in pages or right not in pages:
                continue
            if i == left:
                ordr[left].add(right)
    sorted_rules = sorted(ordr.items(), key=lambda x: len(x[1]), reverse=True)
    result = [i[0] for i in sorted_rules]
    result.extend(sorted_rules[-1][1])
    return result


def page_sorter(data: list[list[str]]) -> tuple[int, int]:
    """
    >>> print(page_sorter(parsers.blocks('test.txt')))
    (143, 123)"""
    ordering, page_numbers = data
    ordering = [tuple(map(int, line.split('|'))) for line in ordering]
    pages = [list(map(int, line.split(','))) for line in page_numbers]
    part1 = part2 = 0
    for line in pages:
        correct_order = sort_pages(ordering, line)
        mid = correct_order[len(correct_order) // 2]
        if line == correct_order:
            part1 += mid
        else:
            part2 += mid
    return part1, part2


print(page_sorter(parsers.blocks(loader.get())))  # 4996, 6311
