with open('input01.txt') as f:
    data = f.read().splitlines()

inv = [[]]
for i in data:
    if i == '':
        inv.append([])
    else:
        i = int(i)
        inv[-1].append(i)

sums = []
for i in inv:
    total = sum(i)
    sums.append(total)

print(max(sums))  # 69883

sums.sort(reverse=True)

print(sum(sums[0:3]))  # 207576
