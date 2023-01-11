with open('input01.txt') as f:
    data = f.read()

sums = [sum(int(i) for i in line.split('\n') if i != '') for line in data.split('\n\n')]

print(max(sums))  # 69883

sums.sort(reverse=True)

print(sum(sums[0:3]))  # 207576
