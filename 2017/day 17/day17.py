from tools import loader, parsers


def part_1(steps: str) -> int:
    """
    >>> print(part_1('3'))
    638"""
    spinlock = [0]
    index = 0
    for i in range(1, 2018):
        index = (index + int(steps)) % len(spinlock) + 1
        spinlock.insert(index, i)
    return spinlock[(index + 1) % len(spinlock)]


def part_2(steps: str) -> int:
    index = 0
    result = 0
    for num in range(1, 50_000_001):
        # next index after zero
        if (index + int(steps)) % num + 1 == 1:
            result = num
    return result


print(part_1(parsers.string(loader.get())))  # 355
print(part_2(parsers.string(loader.get())))  # 6154117
