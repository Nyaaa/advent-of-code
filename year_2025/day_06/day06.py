import numpy as np

from tools import loader, parsers


def part1(data: list[str]) -> int:
    """
    >>> print(part1(parsers.lines('test.txt')))
    4277556
    """
    arr = np.genfromtxt(data, dtype=str)
    mask = arr[-1, :] == '+'
    values = arr[:-1, :].astype(int)
    sums = values[:, mask].sum(axis=0)
    products = values[:, ~mask].prod(axis=0)
    return np.sum(sums) + np.sum(products)


def part2(data: list[str]) -> int:
    """
    >>> print(part2(parsers.lines('test.txt', strip=False)))
    3263827
    """
    result = 0
    arr = np.genfromtxt(data, delimiter=1, dtype=str)
    split_indices = [i + 1 for i in np.flatnonzero(np.all(arr == ' ', axis=0))]
    sub_arrays = np.hsplit(arr, split_indices)

    for problem in sub_arrays:
        vals_str = np.char.strip(np.apply_along_axis(''.join, axis=0, arr=problem[:-1]))
        vals = vals_str[vals_str != ''].astype(int)
        result += np.sum(vals) if problem[-1:, 0] == '+' else np.prod(vals)
    return result


print(part1(parsers.lines(loader.get())))  # 6295830249262
print(part2(parsers.lines(loader.get(), strip=False)))  # 9194682052782
