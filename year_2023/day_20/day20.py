import re
from collections import defaultdict, deque
from itertools import cycle
from math import prod

from tools import loader, parsers


class Module:
    def __init__(self, name: str, targets: list[str]) -> None:
        self.name = name
        self.targets = targets

    @staticmethod
    def process(signal: bool, source: str) -> list:  # noqa: ARG004
        return []


class Broadcaster(Module):
    def process(self, signal: bool, source: str) -> list[tuple[str, bool, str]]:  # noqa: ARG002
        return [(self.name, signal, i) for i in self.targets]


class Flipper(Module):
    def __init__(self, name: str, targets: list[str]) -> None:
        super().__init__(name, targets)
        self.state = cycle([True, False])

    def process(self, signal: bool, source: str) -> list[tuple[str, bool, str]]:  # noqa: ARG002
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


def signals(data: list[str]) -> int:
    """
    >>> print(signals(parsers.lines('test.txt')))
    32000000
    >>> print(signals(parsers.lines('test2.txt')))
    11687500"""
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

    def push_button(times: int) -> dict[bool, int]:
        pulses = {True: 0, False: 0}
        for _ in range(times):
            queue = deque([('broadcaster', False, 'broadcaster')])
            while queue:
                source, signal, target = queue.popleft()
                if source == target:
                    pulses[False] += 1
                module = modules.get(target)
                if not module:
                    module = Module(name=target, targets=[])
                    modules[target] = module
                n = module.process(signal=signal, source=source)
                for new_signal in n:
                    queue.append(new_signal)
                    pulses[new_signal[1]] += 1
        return pulses

    out = push_button(1000)
    return prod(out.values())


print(signals(parsers.lines(loader.get())))  # 806332748
