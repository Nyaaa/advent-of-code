import logging
import re
from itertools import count

from tools import loader, parsers, timer


class Plants:
    def __init__(self, data: list[list[str]]) -> None:
        self.seeds = list(map(int, re.findall(r'\d+', data[0][0])))
        self.tables = []
        for block in data[1:]:
            _block = []
            for line in block[1:]:
                nums = tuple(map(int, line.split()))
                destination = range(nums[0], nums[0] + nums[2])
                source = range(nums[1], nums[1] + nums[2])
                _block.append((source, destination))
            self.tables.append(_block)

    def part_1(self) -> int:
        """
        >>> print(Plants(parsers.blocks('test.txt')).part_1())
        35"""
        result = float('inf')
        for seed in self.seeds:
            current = seed
            for table in self.tables:
                match = None
                for line in table:
                    if current in line[0]:
                        match = line[1].start + current - line[0].start
                current = match or current
            result = min(result, current)
        return result

    @timer.wrapper
    def part_2(self) -> int:
        """Bruteforce, takes a while.
        >>> print(Plants(parsers.blocks('test.txt')).part_2())
        46"""
        a = list(zip(self.seeds[::2], self.seeds[1::2]))
        seeds = [range(i[0], i[0] + i[1]) for i in a]

        for seed in count(1):
            if seed % 1_000_000 == 0:
                logging.debug(seed)
            current = seed
            for table in self.tables[::-1]:
                match = None
                for line in table:
                    if current in line[1]:
                        match = line[0].start + current - line[1].start
                current = match or current
            if any(current in s for s in seeds):
                return seed
        raise ValueError('Solution not found')


print(Plants(parsers.blocks(loader.get())).part_1())  # 484023871
print(Plants(parsers.blocks(loader.get())).part_2())  # 46294175
