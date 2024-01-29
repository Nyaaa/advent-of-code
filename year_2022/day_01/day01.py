from tools import loader, parsers


def puzzle(data: list[list[str]]) -> tuple[int, int]:
    """
    >>> puzzle(parsers.blocks('test.txt'))
    (24000, 45000)
    """
    sums = sorted(sum(int(i) for i in block) for block in data)
    return sums[-1], sum(sums[-3:])


print(puzzle(parsers.blocks(loader.get())))  # 69883, 207576
