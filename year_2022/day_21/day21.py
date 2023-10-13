import operator
import sys
from collections import deque

from tools import loader, parsers


class Monkey:
    def __init__(self, data: list[str]) -> None:
        self.known = {}
        self.unknown = {}

        for line in data:
            line = line.replace(':', '').split()
            if len(line) == 2:
                self.known[line[0]] = int(line[1])
            else:
                self.unknown[line[0]] = [*line[1:]]

    @staticmethod
    def operation(a: int, op: str, b: int) -> int:
        match op:
            case '+':
                result = operator.add(a, b)
            case '-':
                result = operator.sub(a, b)
            case '*':
                result = operator.mul(a, b)
            case '/':
                result = operator.truediv(a, b)
            case _:
                raise KeyError
        return result

    def part_1(self) -> int:
        """test part 1:
        >>> print(Monkey(parsers.lines('test.txt')).part_1())
        152"""
        iterate = deque(self.unknown)
        known = self.known.copy()
        while iterate:
            i = iterate.popleft()
            m1, op, m2 = self.unknown[i]
            try:
                known[i] = self.operation(known[m1], op, known[m2])
            except KeyError:
                iterate.append(i)
        return int(known['root'])

    def part_2(self) -> int:
        """Binary search
        test part 2:
        >>> print(Monkey(parsers.lines('test.txt')).part_2())
        301"""
        self.unknown['root'][1] = '-'
        left = 0
        right = sys.maxsize
        modifier = self.part_1() > 0

        while True:
            middle = (left + right) // 2
            self.known['humn'] = middle
            diff = self.part_1()

            if (diff > 0 and modifier) or (diff < 0 and not modifier):
                left = middle + 1
            elif (diff > 0 and not modifier) or (diff < 0 and modifier):
                right = middle - 1
            else:
                return middle


print(Monkey(parsers.lines(loader.get())).part_1())  # 194058098264286
print(Monkey(parsers.lines(loader.get())).part_2())  # 3592056845086
