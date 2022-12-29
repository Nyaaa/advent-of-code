from itertools import zip_longest
from tools import parsers


def compare(left, right) -> bool:
    if not isinstance(left, list): left = [left]
    if not isinstance(right, list): right = [right]

    zipped = list(zip_longest(left, right))
    for i, j in zipped:
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


def flatten(list_of_lists: list) -> list:
    if len(list_of_lists) == 0:
        return list_of_lists
    if isinstance(list_of_lists[0], list):
        if len(list_of_lists[0]) == 0:
            list_of_lists[0] = [0]
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list_of_lists[:1] + flatten(list_of_lists[1:])


# part 1
data = parsers.blocks('input13.txt')
result = {True: [], False: []}
for index in range(1, len(data) + 1):
    c_left, c_right = data[index - 1]
    c_left, c_right = eval(c_left), eval(c_right)
    res = compare(c_left, c_right)
    result[res].append(index)
print(sum(result[True]))  # 5252

# part 2
data = parsers.lines('input13.txt')
data.append('[[2]]')
data.append('[[6]]')
sort = [flatten(eval(line)) for line in data if line != '']
sort.sort()
print((sort.index([2]) + 1) * (sort.index([6]) + 1))  # 20592
