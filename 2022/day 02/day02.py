from tools import parsers, loader

moves = {'A': 1, 'X': 1,  # rock
         'B': 2, 'Y': 2,  # paper
         'C': 3, 'Z': 3}  # scissors
wins = [(3, 1), (1, 2), (2, 3)]
res = {'X': 'loss', 'Y': 'draw', 'Z': 'win'}
test = ['A Y', 'B X', 'C Z']
data = parsers.lines(loader.get())


def check_win(i: int, j: int):
    if i == j:
        return 'draw', j + 3
    elif (i, j) in wins:
        return 'win', j + 6
    else:
        return 'loss', j + 0


def part_1(data):
    """test part 1:
    >>> print(part_1(test))
    15"""
    part1 = 0
    for round in data:
        i = moves[round[0]]
        j = moves[round[2]]
        outcome, result = check_win(i, j)
        part1 += result
    return part1


def part_2(data):
    """test part 2:
    >>> print(part_2(test))
    12"""
    part2 = 0
    for round in data:
        i = moves[round[0]]
        outcome = round[2]
        for n in 1, 2, 3:
            if check_win(i, n)[0] == res[outcome]:
                part2 += check_win(i, n)[1]
    return part2


print(part_1(data))  # 11603
print(part_2(data))  # 12725
