import re
from typing import NamedTuple

from tools import parsers, loader


class Rule(NamedTuple):
    name: str
    set1: tuple[int, int]
    set2: tuple[int, int]

    def value_is_valid(self, value: int) -> bool:
        a = self.set1[0] <= value <= self.set1[1]
        b = self.set2[0] <= value <= self.set2[1]
        return any((a, b))


class Ticket:
    def __init__(self, data: list):
        self.rules = []
        for rule in data[0]:
            name = rule.split(':')[0]
            r = [int(i) for i in re.findall(r'\d+', rule)]
            self.rules.append(Rule(name, (r[0], r[1]), (r[2], r[3])))
        self.own_ticket = [int(i) for i in data[1][1].split(',')]
        self.tickets = [[int(i) for i in t.split(',')] for t in data[2][1:]]

    def part_1(self):
        """
        >>> print(Ticket(parsers.blocks('test.txt')).part_1())
        71"""
        invalid = []
        invalid_tickets = []
        for ticket in self.tickets:
            for val in ticket:
                if not any([rule.value_is_valid(val) for rule in self.rules]):
                    invalid.append(val)
                    invalid_tickets.append(ticket)
        self.tickets = [t for t in self.tickets if t not in invalid_tickets]
        return sum(invalid)

    def part_2(self):
        self.part_1()
        return self.tickets


# print(Ticket(parsers.blocks(loader.get())).part_1())  # 20231
print(Ticket(parsers.blocks('test.txt')).part_2())