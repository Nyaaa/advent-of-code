test = ['30373',
        '25512',
        '65332',
        '33549',
        '35390']
data = test

# with open('input08.txt') as f:
#     data = f.read().splitlines()


grid = []
for row in data:
    row = [int(i) for i in row]
    grid.append(row)

# print(grid)

x = len(grid[0])
y = len(grid)


def check(tree, _x, _y):
    # print(tree, grid[_y], _x, _y)
    # detecting edges
    if _x  == 0 or _x == x-1:
        return True
    elif _y == 0 or _y == y-1:
        return True
    else:
        return False


vis = []

for _x in range(x):
    row = []
    for _y in range(y):
        tree = grid[_x][_y]
        visible = check(tree, _x, _y)
        row.append(visible)
        # print(tree, visible)
    vis.append(row)

for row in vis:
    print(row)
