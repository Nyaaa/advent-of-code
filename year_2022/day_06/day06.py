from tools import loader, parsers


def puzzle(data: str, length: int) -> int:
    """ Rolling window, 1D array

    test part 1:
    >>> print(puzzle('zcfzfwzzqfrljwzlrfnpqdbhtmscgvjw', 4))
    11

    test part 2:
    >>> print(puzzle('zcfzfwzzqfrljwzlrfnpqdbhtmscgvjw', 14))
    26
    """
    for i in range(len(data) - length):
        chunk = data[i: i + length]
        if len(set(chunk)) == len(chunk):
            return i + length
    raise ValueError('No solution found')


print(puzzle(parsers.string(loader.get()), 4))  # 1779
print(puzzle(parsers.string(loader.get()), 14))  # 2635
