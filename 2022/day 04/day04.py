from tools import parsers, loader
d = parsers.lines(loader.get())
test = ['2-4,6-8', '2-3,4-5', '5-7,7-9', '2-8,3-7', '6-6,4-6', '2-6,4-8']


def get_task(_range):
    start, stop = map(int, _range.split('-'))
    return list(range(start, stop + 1))


def puzzle(data, part: int):
    """test part 1:
    >>> print(puzzle(test, 1))
    2

    test part 2:
    >>> print(puzzle(test, 2))
    4"""

    result = 0
    for pair in data:
        team = pair.split(',')
        pair = [get_task(elf) for elf in team]

        # one list contains another
        if (set(pair[0]).issubset(pair[1]) or set(pair[1]).issubset(pair[0])) and part == 1:
            result += 1

        # partial overlap
        if list(set(pair[0]).intersection(pair[1])) and part == 2:
            result += 1
    return result


print(puzzle(d, 1))  # 456
print(puzzle(d, 2))  # 808
