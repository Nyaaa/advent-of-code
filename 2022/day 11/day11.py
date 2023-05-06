from tools import parsers, loader
import re

d = parsers.blocks(loader.get())
t = parsers.blocks('test11.txt')


class Monke:
    def __init__(self, _id: int):
        self.id: int = _id
        self.inventory: list[int] = []
        self.test: int = 0
        self.target_true: int = 0
        self.target_false: int = 0
        self.op: str = ''
        self.op_val: str = ''
        self.inspected: int = 0

    def action(self, relief: bool, monkeys: list, modulo: int):
        for i, item in enumerate(self.inventory):
            self.inspected += 1

            try:
                value = int(self.op_val)
            except ValueError:
                value = item

            if self.op == '*':
                item *= value
            else:
                item += value

            if relief:
                item //= 3
            else:
                item %= modulo

            self.inventory[i] = item

        for item in self.inventory:
            if item % self.test == 0:
                monkeys[self.target_true].inventory.append(item)
            else:
                monkeys[self.target_false].inventory.append(item)
        self.inventory.clear()


class Main:
    def __init__(self, data):
        self.monkeys = []
        self.modulo = 1

        for monke in data:
            index = int(re.search(r'\d+', monke[0]).group())
            self.monkeys.append(Monke(index))
            self.monkeys[index].inventory = [int(i) for i in re.findall(r'\d+', monke[1])]
            self.monkeys[index].op, self.monkeys[index].op_val = \
                re.search(r'([+*]) ([A-Za-z0-9]+)', monke[2]).group().split()
            self.monkeys[index].test = int(re.search(r'\d+', monke[3]).group())
            self.monkeys[index].target_true = int(re.search(r'\d+', monke[4]).group())
            self.monkeys[index].target_false = int(re.search(r'\d+', monke[5]).group())

        for monkey in self.monkeys:
            self.modulo *= monkey.test

    def start(self, rounds: int, relief: bool):
        """test part 1:
        >>> print(Main(t).start(20, True))
        10605

        test part 2:
        >>> print(Main(t).start(10000, False))
        2713310158"""
        _round = 0
        while _round < rounds:
            _round += 1
            for monkey in self.monkeys:
                monkey.action(relief, self.monkeys, self.modulo)

        activity = [monkey.inspected for monkey in self.monkeys]
        activity.sort()
        return activity[-1] * activity[-2]


# part 1
print(Main(d).start(20, True))  # 78678

# part 2
print(Main(d).start(10000, False))  # 15333249714
