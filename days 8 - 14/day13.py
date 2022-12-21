from tools import parsers
from itertools import zip_longest

data = parsers.blocks('input13.txt')

def compare(left, right, out: bool = True):
    if not isinstance(left, list): left = [left]
    if not isinstance(right, list): right = [right]

    zipped = list(zip_longest(left, right))
    print(zipped)
    for i in zipped:
        print(i)
        if i[1] is None: return False
        elif i[0] is None: return True
        elif isinstance(i[0], int) and isinstance(i[1], int):
            if i[0] > i[1]: return False
        else:
            return compare(i[0], i[1], out)
    return out


result = {True: [], False: []}
for chunk in range(1, len(data) + 1):
    c_left, c_right = data[chunk-1]
    c_left = eval(c_left)
    c_right = eval(c_right)
    res = compare(c_left, c_right)
    result[res].append(chunk)
    print('chunk', chunk, res)

print(sum(result[True]))  # fails