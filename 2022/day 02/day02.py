moves = {'A': 1, 'X': 1,  # rock
         'B': 2, 'Y': 2,  # paper
         'C': 3, 'Z': 3}  # scissors
wins = [(3, 1), (1, 2), (2, 3)]
res = {'X': 'loss', 'Y': 'draw', 'Z': 'win'}

with open('input02.txt') as f:
    data = f.read().splitlines()


def check_win(i: int, j: int):
    if i == j:
        return 'draw', j + 3
    elif (i, j) in wins:
        return 'win', j + 6
    else:
        return 'loss', j + 0

# part 1

part1 = 0
for round in data:
    i = moves[round[0]]
    j = moves[round[2]]
    outcome, result = check_win(i, j)
    part1 += result
    # print(i, j, result, outcome)

print(part1)  # 11603

# part 2

part2 = 0
for round in data:
    i = moves[round[0]]
    outcome = round[2]
    for n in 1, 2, 3:
        if check_win(i, n)[0] == res[outcome]:
            part2 += check_win(i, n)[1]
    # print(i, outcome, play)

print(part2)  # 12725
