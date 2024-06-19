import re

from tools import loader, parsers

DATA = parsers.blocks(loader.get())

molecule = DATA[1][0][::-1]
replacements = dict(line[::-1].split(' >= ') for line in DATA[0])


part1 = set()
for value, key in replacements.items():
    part1.update(f'{molecule[:match.start()]}{value}{molecule[match.end():]}'
                 for match in re.finditer(key, molecule))

print(len(part1))  # 576


part2 = 0
subs = re.compile('|'.join(replacements))
while molecule != 'e':
    molecule = re.sub(subs, lambda x: replacements[x.group()], molecule, count=1)
    part2 += 1

print(part2)  # 207
