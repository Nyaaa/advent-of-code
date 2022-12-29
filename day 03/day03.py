import string

letters = list(string.ascii_letters)
numbers = list(range(1, 53))
priority = dict(zip(letters, numbers))

with open('input03.txt') as f:
    data = f.read().splitlines()

# part 1


def common(one, *rest):
    return list(set(one).intersection(*rest))


value = 0
for rucksack in data:
    mid = len(rucksack) // 2
    pocket1 = list(rucksack[:mid])
    pocket2 = list(rucksack[mid:])
    item = common(pocket1, pocket2)
    value += priority[item[0]]

print(value)

# part 2

value = 0
for i in range(0, len(data), 3):
    a, b, c = data[i], data[i + 1], data[i + 2]
    item = common(a, b, c)
    value += priority[item[0]]

print(value)
