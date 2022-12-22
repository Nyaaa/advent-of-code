from tools import parsers
from itertools import zip_longest

data = parsers.blocks('input13.txt')


def compare(left, right):
    if not isinstance(left, list): left = [left]
    if not isinstance(right, list): right = [right]

    zipped = list(zip_longest(left, right))
    # print(zipped)
    for i, j in zipped:
        # print(i, j)
        if j is None:
            return False
        elif i is None:
            return True

        if isinstance(i, int) and isinstance(j, int):
            if i != j:
                return i < j
        else:
            out = compare(i, j)
            if out is not None:
                return out


result = {True: [], False: []}
for index in range(1, len(data) + 1):
    c_left, c_right = data[index - 1]
    c_left, c_right = eval(c_left), eval(c_right)
    res = compare(c_left, c_right)
    result[res].append(index)
    # print('index', index, res)

print(result)
print(sum(result[True]))  # 5252
