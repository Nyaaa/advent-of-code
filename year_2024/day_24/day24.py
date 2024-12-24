import operator
import re
from collections import deque
from typing import NamedTuple

import networkx as nx

from tools import loader, parsers


class Signal(NamedTuple):
    a: str
    op: str
    b: str
    out: str

    def operation(self) -> operator:
        match self.op:
            case 'AND': return operator.and_
            case 'OR': return operator.or_
            case 'XOR': return operator.xor


def part_1(data: list[list[str]]) -> int:
    """
    >>> print(part_1(parsers.blocks('test.txt')))
    4
    >>> print(part_1(parsers.blocks('test2.txt')))
    2024"""
    values, signals = data
    wires = {}
    for value in values:
        name, val = value.split(': ')
        wires[name] = int(val)

    queue = deque([])
    for signal in signals:
        a, op, b, out = re.findall(r'\w+', signal)
        queue.append(Signal(a, op, b, out))

    while queue:
        signal = queue.popleft()
        if signal.a in wires and signal.b in wires:
            wires[signal.out] = signal.operation()(wires[signal.a], wires[signal.b])
        else:
            queue.append(signal)

    z = {k: v for k, v in wires.items() if k.startswith('z')}
    binary = ''.join(str(z[key]) for key in sorted(z, reverse=True))
    return int(binary, 2)


def part_2(data: list[list[str]]) -> str:
    values, signals = data
    wires = {}
    for value in values:
        name, val = value.split(': ')
        wires[name] = int(val)

    graph = nx.DiGraph()
    for signal in signals:
        a, op, b, out = re.findall(r'\w+', signal)
        graph.add_edge(a, out, label=op)
        graph.add_edge(b, out, label=op)

    sus = set()
    for signal in signals:
        a, op, b, out = re.findall(r'\w+', signal)
        successors = list(graph.successors(out))
        if (
                (out.startswith('z') and op != 'XOR' and out != 'z45')  # noqa: PLR0916
                or (op == 'XOR' and (
                    len(successors) == 1
                    or not any(i.startswith(j) for j in ('x', 'y', 'z') for i in [a, b, out])))
                or (op == 'AND' and 'x00' not in {a, b} and len(successors) > 1)
        ):
            sus.add(out)

    if len(sus) != 8:
        raise ValueError('Solution not found')
    return ','.join(sorted(sus))


print(part_1(parsers.blocks(loader.get())))  # 48063513640678
print(part_2(parsers.blocks(loader.get())))  # hqh,mmk,pvb,qdq,vkq,z11,z24,z38
