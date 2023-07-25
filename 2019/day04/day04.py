from collections import Counter
from tools import parsers, loader


def validate(password: int, part2: bool) -> bool:
    string = str(password)
    previous = 0
    double = False
    if part2 and 2 in Counter(string).values():
        double = True
    for i in (int(j) for j in string):
        if i < previous:
            return False
        elif not part2 and i == previous and not double:
            double = True
        previous = i
    return True if double else False


START, STOP = [int(i) for i in parsers.lines(loader.get())[0].split('-')]
print(sum(validate(i, False) for i in range(START, STOP)))  # 1919
print(sum(validate(i, True) for i in range(START, STOP)))  # 1291
