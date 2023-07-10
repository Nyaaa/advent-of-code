from collections import deque

from tools import parsers, loader


TEST = '389125467'


class Cups:
    def __init__(self, data: list):
        self.circle = deque([int(i) for i in list(data[0])])
        self.full_circle = self.circle.copy()
        self.current_cup = self.circle[0]

    def get_destination(self, cups: list) -> int:
        destination = self.current_cup - 1
        while True:
            if destination < min(self.full_circle):
                destination = max(self.full_circle)
            if destination in cups:
                destination -= 1
            else:
                break
        return destination

    def rotate(self, index: int, condition: int):
        while self.circle[index] != condition:
            self.circle.rotate(1)

    def play(self) -> str:
        """"
        >>> print(Cups(parsers.inline_test(TEST)).play())
        67384529"""

        for _ in range(1, 101):
            self.rotate(0, self.current_cup)
            self.circle.rotate(-1)
            cups = [self.circle.popleft() for _ in range(3)]
            self.rotate(-1, self.get_destination(cups))
            self.circle.extend(cups)
            self.rotate(0, self.current_cup)
            self.current_cup = self.circle[1]
        self.rotate(0, 1)
        self.circle.popleft()
        return ''.join(str(i) for i in self.circle)


print(Cups(parsers.lines(loader.get())).play())  # 97342568
