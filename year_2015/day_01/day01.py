from itertools import accumulate, dropwhile

from tools import loader, parsers

data = [1 if val == '(' else -1 for val in parsers.string(loader.get())]

print(sum(data))  # 232
print(next(dropwhile(lambda x: x[1] >= 0, enumerate(accumulate(data), start=1)))[0])  # 1783
