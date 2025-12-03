from tools import loader, parsers


def find_max_joltage(line: str, batteries: int) -> int:
    activated = ''
    while batteries > 1:
        batteries -= 1
        maximum = max(line[:-batteries])
        activated += maximum
        line = line[line.index(maximum) + 1:]
    activated += max(line)
    return int(activated)


def battery(data: list[str], batteries: int) -> int:
    """
    >>> print(battery(parsers.lines('test.txt'), batteries=2))
    357
    >>> print(battery(parsers.lines('test.txt'), batteries=12))
    3121910778619
    """
    return sum(find_max_joltage(line, batteries) for line in data)


print(battery(parsers.lines(loader.get()), batteries=2))  # 17244
print(battery(parsers.lines(loader.get()), batteries=12))  # 171435596092638
