import re
from collections import defaultdict, deque
from itertools import count, cycle
from math import lcm, prod

from tools import loader, parsers


class Module:
    def __init__(self, name: str, targets: list[str]) -> None:
        self.name = name
        self.targets = targets

    @staticmethod
    def process(*_: str | bool) -> list:
        return []


class Broadcaster(Module):
    def process(self, signal: bool, _: str) -> list[tuple[str, bool, str]]:
        return [(self.name, signal, i) for i in self.targets]


class Flipper(Module):
    def __init__(self, name: str, targets: list[str]) -> None:
        super().__init__(name, targets)
        self.state = cycle([True, False])

    def process(self, signal: bool, _: str) -> list[tuple[str, bool, str]]:
        if signal is False:
            output = next(self.state)
            return [(self.name, output, i) for i in self.targets]
        return []


class Conjunction(Module):
    def __init__(self, name: str, targets: list[str]) -> None:
        super().__init__(name, targets)
        self.inputs = defaultdict(bool)

    def process(self, signal: bool, source: str) -> list[tuple[str, bool, str]]:
        self.inputs[source] = signal
        out = not all(self.inputs.values())
        return [(self.name, out, i) for i in self.targets]


def signals(data: list[str]) -> tuple[int, int]:
    module_types = {'b': Broadcaster, '%': Flipper, '&': Conjunction}
    modules = {}
    for line in data:
        _from, *_to = re.findall(r'[%&]?\w+', line)
        name = _from if _from == 'broadcaster' else _from[1:]
        modules[name] = module_types[_from[0]](name=name, targets=_to)

    string = ' '.join(data)
    conj = re.findall(r'&\w+', string)
    for i in conj:
        name = i[1:]
        inputs = re.findall(rf'(\w+) -> {name}', string)
        for j in inputs:
            modules[name].inputs[j] = False

    joint = re.findall(r'(\w+) -> rx', string)[0]  # assuming rx has 1 input
    joint_inputs = re.findall(rf'(\w+) -> {joint}', string)
    part_1 = part_2 = 0
    pulses = {True: 0, False: 0}
    high_pulses = {n: 0 for n in joint_inputs}

    for press in count(1):
        if press == 1001:
            part_1 = prod(pulses.values())
        if all(high_pulses.values()):
            part_2 = lcm(*high_pulses.values())
        if part_1 and part_2:
            break
        queue = deque([('broadcaster', False, 'broadcaster')])
        while queue:
            source, signal, target = queue.popleft()
            if source == target:
                pulses[False] += 1
            if source in joint_inputs and signal:
                high_pulses[source] = press
            module = modules.get(target)
            if not module:
                module = Module(name=target, targets=[])
                modules[target] = module
            for new_signal in module.process(signal, source):
                queue.append(new_signal)
                pulses[new_signal[1]] += 1

    return part_1, part_2


print(signals(parsers.lines(loader.get())))  # 806332748, 228060006554227
