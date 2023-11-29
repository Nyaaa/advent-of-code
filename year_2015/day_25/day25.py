import re

from tools import loader, parsers

ROW_TARGET, COL_TARGET = map(int, re.findall(r'\d+', parsers.string(loader.get())))

code = 20151125

iterations = sum(range(ROW_TARGET + COL_TARGET - 1)) + COL_TARGET

for _ in range(iterations - 1):
    code = code * 252533 % 33554393

print(code)  # 9132360
