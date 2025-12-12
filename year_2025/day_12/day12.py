from math import prod

from tools import loader, parsers


def fit_present(data: list[list[str]]) -> int:
    """
    >>> print(fit_present(parsers.blocks('test.txt')))
    2
    """
    *shapes, regions = data
    shape_areas = [sum(1 if j == '#' else 0 for i in s[1:] for j in i) for s in shapes]
    result = 0
    for region in regions:
        r_area, r_shapes = region.split(': ')
        area = prod([int(i) for i in r_area.split('x')])
        area_needed = sum(shape_areas[i] * int(j) for i, j in enumerate(r_shapes.split(' ')))
        if area_needed * 1.2 < area:
            result += 1
    return result


print(fit_present(parsers.blocks(loader.get())))  # 485
