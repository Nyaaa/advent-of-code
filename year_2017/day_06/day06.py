from tools import loader, parsers


def reallocate_mem(data: str, part2: bool) -> int:
    """
    >>> print(reallocate_mem('0\\t2\\t7\\t0', False))
    5

    >>> print(reallocate_mem('0\\t2\\t7\\t0', True))
    4"""
    data = [int(i) for i in data.split('\t')]
    seen = set()
    counter = 0
    while True:
        if s := tuple(data) in seen:
            if not part2:
                break
            seen, counter, part2 = {s}, 0, False
        seen.add(tuple(data))
        highest = max(data)
        index = data.index(highest)
        data[index] = 0
        while highest > 0:
            index = (index + 1) % len(data)
            data[index] += 1
            highest -= 1
        counter += 1
    return counter


print(reallocate_mem(parsers.string(loader.get()), False))  # 6681
print(reallocate_mem(parsers.string(loader.get()), True))  # 2392
