with open('input04.txt') as f:
    data = f.read().splitlines()


def get_task(_range):
    start, stop = map(int, _range.split('-'))
    return list(range(start, stop + 1))


part1 = 0
part2 = 0
for pair in data:
    team = pair.split(',')
    pair = [get_task(elf) for elf in team]

    # part 1
    # one list contains another
    if set(pair[0]).issubset(pair[1]) or set(pair[1]).issubset(pair[0]):
        part1 += 1

    # part 2
    # partial overlap
    if list(set(pair[0]).intersection(pair[1])):
        part2 += 1

print(part1)  # 456
print(part2)  # 808
