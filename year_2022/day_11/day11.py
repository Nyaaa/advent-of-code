import operator
import re
from dataclasses import dataclass

from tools import loader, parsers


@dataclass
class Monke:
    inventory: list[int]
    test: int
    targets: dict[bool, int]
    operation: operator
    op_val: str
    inspected: int = 0

    def action(self, relief: bool, monkeys: list, modulo: int) -> None:
        for item in self.inventory:
            try:
                value = int(self.op_val)
            except ValueError:
                value = item

            item = self.operation(item, value)

            if relief:
                item //= 3
            else:
                item %= modulo

            target = self.targets[item % self.test == 0]
            monkeys[target].inventory.append(item)
        self.inspected += len(self.inventory)
        self.inventory.clear()


def solve(data: list[list[str]], rounds: int, relief: bool) -> int:
    """
    >>> print(solve(parsers.blocks('test11.txt'), 20, True))
    10605
    >>> print(solve(parsers.blocks('test11.txt'), 10000, False))
    2713310158"""
    monkeys = []
    modulo = 1

    for monke in data:
        m = Monke(
            inventory=list(map(int, re.findall(r'\d+', monke[1]))),
            operation=operator.add if monke[2].split()[4] == '+' else operator.mul,
            op_val=monke[2].split()[-1],
            test=int(monke[3].split()[3]),
            targets={True: int(monke[4].split()[5]), False: int(monke[5].split()[5])},
        )
        monkeys.append(m)
        modulo *= m.test

    for _ in range(rounds):
        for monkey in monkeys:
            monkey.action(relief, monkeys, modulo)

    monkeys.sort(key=lambda x: x.inspected, reverse=True)
    return monkeys[0].inspected * monkeys[1].inspected


print(solve(parsers.blocks(loader.get()), rounds=20, relief=True))  # 78678
print(solve(parsers.blocks(loader.get()), rounds=10000, relief=False))  # 15333249714
