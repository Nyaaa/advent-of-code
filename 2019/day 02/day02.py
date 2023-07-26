from tools import parsers, loader, intcode

pc = intcode.Intcode(parsers.lines(loader.get()))


def part_1():
    pc.data[1] = 12
    pc.data[2] = 2
    return pc.run()[0]


def part_2():
    for i in range(100):
        for j in range(100):
            pc.data[1] = i
            pc.data[2] = j
            if pc.run()[0] == 19690720:
                return 100 * i + j
    raise ValueError


print(part_1())  # 5290681
print(part_2())  # 5741
