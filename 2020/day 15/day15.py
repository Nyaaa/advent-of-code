from collections import deque

from tools import loader, parsers


def recite(numbers: str, steps: int) -> int:
    """
    >>> print(recite("0,3,6", 2020))
    436

    >>> print(recite("0,3,6", 30000000))
    175594"""
    numbers = [int(i) for i in numbers.split(',')]
    called_numbers = {val: deque([i + 1], maxlen=2) for i, val in enumerate(numbers)}

    previous_number = called_numbers[numbers[-1]]
    new_value = None
    step = len(numbers)
    while step != steps:
        step += 1
        new_value = 0 if len(previous_number) <= 1 else previous_number[-1] - previous_number[-2]

        if previous_number := called_numbers.get(new_value):
            previous_number.append(step)
        else:
            previous_number = called_numbers[new_value] = deque([step], maxlen=2)

    return new_value


print(recite(parsers.string(loader.get()), 2020))  # 763
print(recite(parsers.string(loader.get()), 30000000))  # 1876406
