from tools import parsers, loader

TEST = '389125467'


class Cups:
    def __init__(self, data: list[str], part2: bool = False):
        data = [int(i) for i in data[0]]
        if part2:
            data.extend(list(range(10, 1_000_001)))
        self.circle = {number: data[(i + 1) % len(data)] for i, number in enumerate(data)}
        self.current_cup = data[0]
        self.min, self.max = min(data), max(data)

    def get_destination(self, cups: list) -> int:
        destination = self.current_cup - 1
        while True:
            if destination < self.min:
                destination = self.max
            if destination in cups:
                destination -= 1
            else:
                break
        return destination

    def play(self, steps: int):
        for _ in range(steps):
            cups = [cup1 := self.circle[self.current_cup],
                    cup2 := self.circle[cup1],
                    self.circle[cup2]]
            destination = self.get_destination(cups)
            self.circle[self.current_cup] = self.circle[cups[-1]]
            self.current_cup = self.circle[cups[-1]]
            self.circle[cups[-1]] = self.circle[destination]
            self.circle[destination] = cups[0]

    def part_1(self):
        """"
        >>> print(Cups(parsers.inline_test(TEST)).part_1())
        67384529"""
        self.play(100)
        result = []
        step = self.circle[1]
        for _ in self.circle:
            result.append(step)
            step = self.circle[step]
        return ''.join(str(i) for i in result[:-1])

    def part_2(self):
        """"
        >>> print(Cups(parsers.inline_test(TEST), True).part_2())
        149245887792"""
        self.play(10_000_000)
        cup1 = self.circle[1]
        return cup1 * self.circle[cup1]


print(Cups(parsers.lines(loader.get())).part_1())  # 97342568
print(Cups(parsers.lines(loader.get()), True).part_2())  # 902208073192
