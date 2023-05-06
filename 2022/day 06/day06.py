from tools import parsers, loader
d = parsers.lines(loader.get())[0]


def puzzle(data: str, length: int):
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


print(puzzle(d, 4))  # 1779
print(puzzle(d, 14))  # 2635
