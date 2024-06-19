import queue
import re
from collections.abc import Generator

from tools import loader, parsers


class Crane:
    def __init__(self, data: list[list[str]]) -> None:
        self.data = data[1]
        self.stack_list = []
        load = [''.join(s) for s in zip(*data[0][:-1], strict=True)]
        stacks = list(filter(None, (re.findall(r'\w+', i.strip()) for i in load)))
        self.size = len(stacks)
        for i in range(self.size):
            self.stack_list.append(queue.LifoQueue())
            for crate in stacks[i][0][::-1]:
                self.stack_list[i].put(crate)

    def get_moves(self) -> Generator[tuple[int, int, int]]:
        for move in self.data:
            move = move.split(' ')
            yield int(move[1]), int(move[3]), int(move[5])

    def start(self, part2: bool) -> str:
        """
        >>> print(Crane(parsers.blocks('test.txt')).start(False))
        CMZ

        >>> print(Crane(parsers.blocks('test.txt')).start(True))
        MCD"""
        for amount, _from, _to in self.get_moves():
            crates = [self.stack_list[_from - 1].get() for _ in range(amount)]
            if part2:
                crates.reverse()
            for crate in crates:
                self.stack_list[_to - 1].put(crate)
        return ''.join(stack.get() for stack in self.stack_list)


print(Crane(parsers.blocks(loader.get())).start(part2=False))  # PSNRGBTFT
print(Crane(parsers.blocks(loader.get())).start(part2=True))  # BNTZFPMMW
