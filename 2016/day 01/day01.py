from tools import parsers, loader


def part_1(data):
    direction = 1j
    location = 0j
    for i in data.split(', '):
        direction *= -1j if i[0] == 'R' else 1j
        location += direction * int(i[1:])
    return int(abs(location.real) + abs(location.imag))


def part_2(data):
    direction = 1j
    location = 0j
    seen = set()
    for i in data.split(', '):
        direction *= -1j if i[0] == 'R' else 1j
        for _ in range(int(i[1:])):
            location += direction
            if location in seen:
                return int(abs(location.real) + abs(location.imag))
            seen.add(location)
    raise ValueError('Intersection not found')


print(part_1(parsers.string(loader.get())))  # 234
print(part_2(parsers.string(loader.get())))  # 113
