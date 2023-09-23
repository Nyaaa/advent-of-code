from collections.abc import Generator
from tools import parsers, loader


class Bridge:
    def __init__(self, data: list[str]) -> None:
        self.pieces = set(tuple(map(int, line.split('/'))) for line in data)

    def run(self, bridge: set[tuple[int, ...]], port: int) -> Generator[set[tuple[int, int]]]:
        for piece in (i for i in self.pieces if port in i and i not in bridge):
            _bridge = bridge.copy()
            _bridge.add(piece)
            yield from self.run(_bridge, piece[1] if piece[0] == port else piece[0])
        yield bridge

    def start(self) -> tuple[int, int]:
        """
        >>> print(Bridge(parsers.lines('test.txt')).start())
        (31, 19)"""
        bridges = list(self.run(set(), 0))
        longest = []
        length = 0
        for i in bridges:
            if len(i) > length:
                longest = [i]
                length = len(i)
            elif len(i) == length:
                longest.append(i)
        part1 = max(sum(sum(j) for j in i) for i in bridges)
        part2 = max(sum(sum(j) for j in i) for i in longest)
        return part1, part2


print(Bridge(parsers.lines(loader.get())).start())  # 1656, 1642
