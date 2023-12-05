import re

from tools import loader, parsers


def planting(data: list[list[str]]) -> int:
    """
    >>> print(planting(parsers.blocks('test.txt')))
    35"""
    seeds = list(map(int, re.findall(r'\d+', data[0][0])))
    tables = []
    for block in data[1:]:
        _block = []
        for line in block[1:]:
            nums = tuple(map(int, line.split()))
            destination = range(nums[0], nums[0] + nums[2])
            source = range(nums[1], nums[1] + nums[2])
            _block.append((source, destination))
        tables.append(_block)
        pass
    results = []
    for seed in seeds:
        current = seed
        for table in tables:
            match = None
            for line in table:
                if current in line[0]:
                    diff = current - line[0].start
                    match = line[1].start + diff
                    pass
            current = match or current
            pass
        results.append(current)
    return min(results)


print(planting(parsers.blocks(loader.get())))  # 484023871

