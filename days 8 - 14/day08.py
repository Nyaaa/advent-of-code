from itertools import chain
from collections import Counter

test = ['30373', '25512', '65332', '33549', '35390']
# data = test

with open('input08.txt') as f:
    data = f.read().splitlines()


grid = []
for row in data:
    row = [int(i) for i in row]
    grid.append(row)

# print(grid)

x = len(grid[0])
y = len(grid)


def is_visible(_tree, _x, _y):
    # getting edges
    if _x == 0 or _x == x-1 or _y == 0 or _y == y-1:
        return 'Visible'

    # getting rows
    left = grid[_y][:_x]
    right = grid[_y][_x+1:]

    # getting columns
    up = [grid[i][_x] for i in range(0, _y)]
    down = [grid[i][_x] for i in range(_y+1, y)]

    if _tree > max(left) or _tree > max(right) or _tree > max(up) or _tree > max(down):
        return 'Visible'

    return 'Hidden'


vis = []
score = []

for _x in range(x):
    vis_row = []
    for _y in range(y):
        tree = grid[_x][_y]
        visible = is_visible(tree, _y, _x)
        vis_row.append(visible)
    vis.append(vis_row)

# part 1

result = list(chain(*vis))
print(Counter(result))  # 1543
