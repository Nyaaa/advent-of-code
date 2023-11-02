from itertools import combinations
from math import prod

from tools import loader, parsers

DATA = parsers.lines(loader.get())

print(sum(2 * sum(sides) + min(sides) for sides in
          (list(map(prod, combinations(map(int, line.split('x')), 2)))
           for line in DATA)))  # 1606483
print(sum(2 * (j[0] + j[1]) + prod(j) for j in
          (sorted(map(int, line.split('x'))) for line in DATA)))  # 3842356
