from itertools import combinations
from math import prod

from tools import loader, parsers

data = parsers.lines(loader.get())

print(sum(2 * sum(sides) + min(sides) for sides in
          (list(map(prod, combinations(map(int, line.split('x')), 2)))
           for line in data)))  # 1606483
print(sum(2 * (j[0] + j[1]) + prod(j) for j in
          (sorted(map(int, i.split('x'))) for i in data)))  # 3842356
