from tools import parsers, loader
from itertools import islice


class Navigation:
    def __init__(self, data: list):
        self.data = iter(int(i) for i in data[0].split())

    def part_1(self):
        """
        >>> print(Navigation(['2 3 0 3 10 11 12 1 1 0 1 99 2 1 1 2']).part_1())
        138"""
        child, meta = islice(self.data, 2)
        child_value = sum(self.part_1() for _ in range(child))
        meta_value = sum(islice(self.data, meta))
        return child_value + meta_value

    def part_2(self):
        """
        >>> print(Navigation(['2 3 0 3 10 11 12 1 1 0 1 99 2 1 1 2']).part_2())
        66"""
        child, meta = islice(self.data, 2)
        meta_value = islice(self.data, meta)
        if child == 0:
            return sum(meta_value)
        else:
            children = {k: self.part_2() for k in range(1, child + 1)}
            return sum(children.get(i, 0) for i in meta_value)


print(Navigation(parsers.lines(loader.get())).part_1())  # 45194
print(Navigation(parsers.lines(loader.get())).part_2())  # 22989
