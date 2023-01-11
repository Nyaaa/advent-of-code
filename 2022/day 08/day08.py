from itertools import chain
from collections import Counter

test = ['30373', '25512', '65332', '33549', '35390']

with open('input08.txt') as f:
    data = f.read().splitlines()

grid = [[int(i) for i in row] for row in data]
x = len(grid[0])
y = len(grid)


def get_score(side, direction):
    """part 2"""

    _score = 0
    if direction == 1:
        side = side[::-1]  # iterating backwards for left & up

    for i in side:
        if i < tree:
            _score += 1
        else:
            _score += 1
            break

    return _score


def is_visible(_tree, _x, _y):
    """part 1"""

    # getting edges
    if _x == 0 or _x == x-1 or _y == 0 or _y == y-1:
        return 'Visible', 0

    # getting rows
    left = grid[_y][:_x]
    right = grid[_y][_x+1:]

    # getting columns
    up = [grid[i][_x] for i in range(0, _y)]
    down = [grid[i][_x] for i in range(_y+1, y)]

    s_right = get_score(right, 0)
    s_left = get_score(left, 1)
    s_up = get_score(up, 1)
    s_down = get_score(down, 0)
    _score = s_right * s_left * s_up * s_down

    if _tree > max(left) or _tree > max(right) or _tree > max(up) or _tree > max(down):
        return 'Visible', _score

    return 'Hidden', _score


vis = []
scores = []

for _x in range(x):
    vis_row = []
    score_row = []
    for _y in range(y):
        tree = grid[_x][_y]
        visible, score = is_visible(tree, _y, _x)
        vis_row.append(visible)
        score_row.append(score)
    vis.append(vis_row)
    scores.append(score_row)

# part 1

result = list(chain(*vis))
print(Counter(result))  # 1543

# part 2

result = list(chain(*scores))
print(max(result))  # 595080
