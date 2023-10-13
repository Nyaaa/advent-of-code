from tools import loader, parsers


class Cups:
    def __init__(self, data: str, part2: bool = False) -> None:
        data = [int(i) for i in data]
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

    def play(self, steps: int) -> None:
        for _ in range(steps):
            cups = [cup1 := self.circle[self.current_cup],
                    cup2 := self.circle[cup1],
                    self.circle[cup2]]
            destination = self.get_destination(cups)
            self.circle[self.current_cup] = self.circle[cups[-1]]
            self.current_cup = self.circle[cups[-1]]
            self.circle[cups[-1]] = self.circle[destination]
            self.circle[destination] = cups[0]

    def part_1(self) -> str:
        """"
        >>> print(Cups('389125467').part_1())
        67384529"""
        self.play(100)
        result = []
        step = self.circle[1]
        for _ in self.circle:
            result.append(step)
            step = self.circle[step]
        return ''.join(str(i) for i in result[:-1])

    def part_2(self) -> int:
        """"
        >>> print(Cups('389125467', True).part_2())
        149245887792"""
        self.play(10_000_000)
        cup1 = self.circle[1]
        return cup1 * self.circle[cup1]


print(Cups(parsers.string(loader.get())).part_1())  # 97342568
print(Cups(parsers.string(loader.get()), True).part_2())  # 902208073192
