from tools import parsers, loader, intcode


def get_reading(x, y):
    pc = intcode.Intcode(parsers.lines(loader.get()))
    return pc.run([x, y])


def part_2():
    x = 0
    y = 100
    while True:
        if get_reading(x, y):
            y_top = y - 99
            if get_reading(x + 99, y_top):
                return 10000 * x + y_top
            else:
                y += 1
        else:
            x += 1


print(sum(get_reading(x, y) for x in range(50) for y in range(50)))  # 141
print(part_2())  # 15641348
