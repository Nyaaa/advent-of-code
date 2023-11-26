import re
from itertools import combinations
from math import prod

from tools import loader, parsers


def balance(compartments: int) -> int:
    data = parsers.string(loader.get())
    packages = list(map(int, re.findall(r'\d+', data)))
    target_weight = sum(packages) // compartments
    for i in range(1, len(packages)):
        valid_weight = [j for j in combinations(packages, i) if sum(j) == target_weight]
        if entanglement := [prod(v) for v in valid_weight]:
            return min(entanglement)
    raise ValueError('Solution not found')


print(balance(compartments=3))  # 11846773891
print(balance(compartments=4))  # 80393059
