import re
from collections import deque

from scipy.optimize import linprog

from tools import loader, parsers


class Machine:
    def __init__(self, schematics: str) -> None:
        lights, buttons, joltages = re.search(r'\[(.*)] (.+?) {(.*)}', schematics).groups()
        self.target = frozenset(i for i, x in enumerate(lights) if x == '#')
        self.buttons = [frozenset(int(i) for i in re.findall(r'\d', b)) for b in buttons.split()]
        self.joltages = tuple(map(int, joltages.split(',')))


def part_1(data: list[str]) -> int:
    """
    >>> print(part_1(parsers.lines('test.txt')))
    7
    """
    machines = [Machine(line) for line in data]
    result = 0
    for machine in machines:
        queue = deque([(0, frozenset())])
        seen = set()
        while queue:
            presses, state = queue.popleft()
            if state == machine.target:
                result += presses
                break
            for button in machine.buttons:
                new_state = state.symmetric_difference(button)
                if new_state not in seen:
                    seen.add(new_state)
                    queue.append((presses + 1, new_state))
    return result


def part_2(data: list[str]) -> int:
    """
    >>> print(part_2(parsers.lines('test.txt')))
    33
    """
    machines = [Machine(line) for line in data]
    result = 0
    for machine in machines:
        c = [1] * len(machine.buttons)
        a_eq = [[i in j for j in machine.buttons]for i in range(len(machine.joltages))]
        r = linprog(c=c, A_eq=a_eq, b_eq=machine.joltages, integrality=1)
        result += r.fun
    return int(result)


print(part_1(parsers.lines(loader.get())))  # 477
print(part_2(parsers.lines(loader.get())))  # 17970
