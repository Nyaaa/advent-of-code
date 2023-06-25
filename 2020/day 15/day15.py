from dataclasses import dataclass, field
from tools import parsers, loader


TEST = "0,3,6"


@dataclass
class Number:
    value: int
    order: list = field(default_factory=list)

    def __eq__(self, other: int):
        return self.value == other

    def is_first(self):
        return len(self.order) <= 1

    def get_new_number(self):
        return self.order[-1] - self.order[-2]


def recite(numbers: list):
    """
    >>> print(recite(parsers.inline_test(TEST)))
    436"""
    numbers = [int(i) for i in numbers[0].split(',')]
    called_numbers = [Number(value=val, order=[i + 1]) for i, val in enumerate(numbers)]

    previous_number = called_numbers[-1]
    step = len(numbers)
    while step != 2020:
        step += 1
        if previous_number.is_first():
            new_value = 0
        else:
            new_value = previous_number.get_new_number()
        num = (x for x in called_numbers if x.value == new_value)
        try:
            n = next(num)
        except StopIteration:
            n = Number(new_value)
            called_numbers.append(n)
        n.order.append(step)
        previous_number = n
    return previous_number.value


print(recite(parsers.lines(loader.get())))  # 763
