from tools import parsers
import numpy as np

test = """498,4 -> 498,6 -> 496,6
503,4 -> 502,4 -> 502,9 -> 494,9
"""
# data = parsers.inline_test(test)
data = parsers.lines('input14.txt')
data = [[[int(i) for i in coord.split(',')] for coord in line.split(' -> ')] for line in data]
left, right = 500, 501
cave = np.zeros((1, 600), dtype=int)
start = 500

for line in data:
    for i in range(len(line)):
        x, y = line[i][0], line[i][1]
        try:
            next_x, next_y = line[i + 1][0], line[i + 1][1]
        except IndexError:
            break
        max_x, min_x, max_y = max(x, next_x), min(x, next_x), max(y, next_y)
        left = min_x if min_x < left else left
        right = max_x if max_x > right else right
        delta_x = x - next_x
        if max_y > len(cave):
            add = max_y + 1 - len(cave)
            newline = np.zeros((add, 600), dtype=int)
            cave = np.append(cave, newline, axis=0)

        if delta_x > 0:
            for _x in range(next_x, x + 1):
                cave[y, _x] = 1
        if delta_x < 0:
            for _x in range(x, next_x + 1):
                cave[y, _x] = 1

        if y - next_y > 0:
            for _y in range(next_y, y + 1):
                cave[_y, x] = 1
        elif y - next_y < 0:
            for _y in range(y, next_y + 1):
                cave[_y, x] = 1


def fall(row: int, column: int):
    if cropped[row][column] not in [1, 8]:
        return fall(row + 1, column)
    else:
        if cropped[row][column - 1] not in [1, 8]:
            return fall(row, column - 1)
        else:
            if cropped[row][column + 1] not in [1, 8]:
                return fall(row, column + 1)
            else:
                cropped[row - 1][column] = 8


np.set_printoptions(threshold=np.inf, linewidth=200)
start = 501 - left
counter = 0
cropped = cave[:, left - 1:right + 1]
while True:
    try:
        fall(0, start)
        counter += 1
    except IndexError:
        break

print(cropped)

# part 1
print(counter)  # 774
