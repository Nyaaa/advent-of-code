from collections import Counter
from functools import cache
from itertools import pairwise

from tools import loader, parsers


class Polymer:
    def __init__(self, data: list[list[str]]) -> None:
        self.template: str = data[0][0]
        self.rules: dict = dict(x.split(' -> ') for x in data[1])
        self.counter: Counter = Counter(self.template)

    @cache
    def count(self, pair: str, steps: int) -> Counter | None:
        if steps == 0:
            return None
        new = self.rules[pair]
        _counter = Counter(new)
        for i in pair[0] + new, new + pair[1]:
            _counter.update(self.count(pair=i, steps=steps - 1))
        return _counter

    def start(self, steps: int) -> int:
        """test part 1:
        >>> print(Polymer(parsers.blocks('test.txt')).start(10))
        1588

        test part 2:
        >>> print(Polymer(parsers.blocks('test.txt')).start(40))
        2188189693529"""
        for pair in pairwise(self.template):
            pair = ''.join(pair)
            self.counter.update(self.count(pair=pair, steps=steps))
        common = self.counter.most_common()
        return common[0][1] - common[-1][1]


print(Polymer(parsers.blocks(loader.get())).start(10))  # 2223
print(Polymer(parsers.blocks(loader.get())).start(40))  # 2566282754493

