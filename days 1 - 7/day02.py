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

score = 0

for round in data:
    i = moves[round[0]]
    j = moves[round[2]]
    outcome, result = check_win(i, j)
    score += result
    # print(i, j, result, outcome)

print(score)  # 11603

# part 2

score = 0


def choose_move(opponent: int, outcome: str):
    needed_res = res[outcome]
    for i in (1, 2, 3):
        test = check_win(opponent, i)[0]
        if test == needed_res:
            return i


for round in data:
    i = moves[round[0]]
    outcome = round[2]
    play = choose_move(i, outcome)
    score += check_win(i, play)[1]
    # print(i, outcome, play)

print(score)  # 12725
