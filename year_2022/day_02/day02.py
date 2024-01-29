from tools import loader, parsers

moves = {'A': 1, 'X': 1,  # rock
         'B': 2, 'Y': 2,  # paper
         'C': 3, 'Z': 3}  # scissors
wins = {(3, 1), (1, 2), (2, 3)}
res = {'X': 0, 'Y': 3, 'Z': 6}
test = ['A Y', 'B X', 'C Z']
reverse_wins = {1: {0: 3, 3: 1, 6: 2}, 2: {0: 1, 3: 2, 6: 3}, 3: {0: 2, 3: 3, 6: 1}}


def check_win(i: int, j: int) -> tuple[int, int]:
    if i == j:
        return 3, j
    if (i, j) in wins:
        return 6, j
    return 0, j


def part_1(data: list[str]) -> int:
    """test part 1:
    >>> print(part_1(test))
    15"""
    return sum(sum(check_win(moves[r[0]], moves[r[2]])) for r in data)


def part_2(data: list[str]) -> int:
    """test part 2:
    >>> print(part_2(test))
    12"""
    return sum(reverse_wins[moves[r[0]]][res[r[2]]] + res[r[2]] for r in data)


print(part_1(parsers.lines(loader.get())))  # 11603
print(part_2(parsers.lines(loader.get())))  # 12725
