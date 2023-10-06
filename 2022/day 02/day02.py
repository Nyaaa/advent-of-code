from tools import loader, parsers

moves = {'A': 1, 'X': 1,  # rock
         'B': 2, 'Y': 2,  # paper
         'C': 3, 'Z': 3}  # scissors
wins = [(3, 1), (1, 2), (2, 3)]
res = {'X': 'loss', 'Y': 'draw', 'Z': 'win'}
test = ['A Y', 'B X', 'C Z']


def check_win(i: int, j: int) -> tuple[str, int]:
    if i == j:
        return 'draw', j + 3
    elif (i, j) in wins:
        return 'win', j + 6
    else:
        return 'loss', j + 0


def part_1(data: list[str]) -> int:
    """test part 1:
    >>> print(part_1(test))
    15"""
    return sum(check_win(moves[r[0]], moves[r[2]])[1] for r in data)


def part_2(data: list[str]) -> int:
    """test part 2:
    >>> print(part_2(test))
    12"""
    part2 = 0
    for r in data:
        i = moves[r[0]]
        for n in 1, 2, 3:
            if check_win(i, n)[0] == res[r[2]]:
                part2 += check_win(i, n)[1]
    return part2


print(part_1(parsers.lines(loader.get())))  # 11603
print(part_2(parsers.lines(loader.get())))  # 12725
