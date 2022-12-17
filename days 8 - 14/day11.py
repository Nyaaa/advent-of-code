FILENAME = 'input11.txt'


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

    def action(self, relief: bool):
        for i in range(len(self.inventory)):
            item = self.inventory[i]
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
                item %= m.modulo

            self.inventory[i] = item

        for item in self.inventory:
            if item % self.test == 0:
                m.monkeys[self.target_true].inventory.append(item)
            else:
                m.monkeys[self.target_false].inventory.append(item)
        self.inventory.clear()


class Main:
    def __init__(self):
        self.monkeys = []
        self.modulo = 1

        with open(FILENAME) as f:
            data = f.read().splitlines()
        index = 0

        for line in data:
            line = line.strip().replace(',', ' ').replace(':', ' ').split()
            if not line:
                continue

            match line[0]:
                case 'Monkey':
                    index = int(line[1])
                    self.monkeys.append(Monke(index))
                case 'Starting':
                    for i in line:
                        try:
                            self.monkeys[index].inventory.append(int(i))
                        except ValueError:
                            continue
                case 'Test':
                    self.monkeys[index].test = int(line[-1])
                case 'If' if line[1] == 'true':
                    self.monkeys[index].target_true = int(line[-1])
                case 'If' if line[1] == 'false':
                    self.monkeys[index].target_false = int(line[-1])
                case 'Operation':
                    self.monkeys[index].op = line[-2]
                    self.monkeys[index].op_val = line[-1]

        # Ugh, math
        for monkey in self.monkeys:
            self.modulo *= monkey.test

    def start(self, rounds: int, relief: bool):
        _round = 0
        while _round < rounds:
            _round += 1
            for monkey in self.monkeys:
                monkey.action(relief)

        activity = []
        for monkey in self.monkeys:
            activity.append(monkey.inspected)
        activity.sort()
        return activity[-1] * activity[-2]


# part 1
m = Main()
print(m.start(20, True))  # 78678

# part 2
m = Main()
print(m.start(10000, False))  # 15333249714
