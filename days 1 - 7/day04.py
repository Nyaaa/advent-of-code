with open('input04.txt') as f:
    data = f.read().splitlines()


def get_task(_range):
    start, stop = _range.split('-')
    start, stop = int(start), int(stop)
    length = stop - start
    sequence = []
    for i in range(0, length + 1):
        sequence.append(start + i)
    return sequence


counter1 = 0
counter2 = 0
for pair in data:
    team = pair.split(',')
    pair = []
    for elf in team:
        pair.append(get_task(elf))

    # part 1
    # one list contains another
    if set(pair[0]).issubset(pair[1]) or set(pair[1]).issubset(pair[0]):
        counter1 += 1

    # part 2
    # partial overlap
    if list(set(pair[0]).intersection(pair[1])):
        counter2 += 1

print(counter1)  # 456
print(counter2)  # 808
