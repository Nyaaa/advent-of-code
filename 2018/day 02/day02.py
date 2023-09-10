from collections import Counter
from itertools import combinations
import difflib

from tools import parsers, loader


def part_1() -> int:
    counts = {2: 0, 3: 0}
    for line in data:
        c = Counter(line).values()
        if 2 in c:
            counts[2] += 1
        if 3 in c:
            counts[3] += 1
    return counts[2] * counts[3]


def part_2() -> str:
    matcher = difflib.SequenceMatcher()
    for a, b in combinations(data, 2):
        matcher.set_seqs(a, b)
        if matcher.ratio() > 0.96:  # 1 char diff in 26 char string
            return ''.join(i[2] for i in difflib.ndiff(a, b) if i[0] == ' ')


data = parsers.lines(loader.get())
print(part_1())  # 6642
print(part_2())  # cvqlbidheyujgtrswxmckqnap
