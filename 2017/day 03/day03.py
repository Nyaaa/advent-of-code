from tools import parsers, loader


def part_1(target):
    target = int(target)
    direction = 1j
    location = 0j
    spiral = set()
    counter = 0
    while counter != target:
        counter += 1
        spiral.add(location)
        location += direction
        if location + direction * 1j not in spiral:
            direction *= 1j
    return int(abs(location.real) + abs(location.imag)) - 1


def part_2(target):
    target = int(target)
    direction = 1j
    location = 0j
    adjacent = [-1-1j, -1+0j, -1+1j,
                0-1j, 0+0j, 0+1j,
                1-1j, 1+0j, 1+1j]
    spiral = dict()
    adj = 0
    while adj <= target:
        adj = sum(spiral.get(location + direction * i, 0) for i in adjacent) or 1
        spiral[location] = adj
        location += direction
        if location + direction * 1j not in spiral:
            direction *= 1j
    return adj


print(part_1(parsers.string(loader.get())))  # 438
print(part_2(parsers.string(loader.get())))  # 266330
