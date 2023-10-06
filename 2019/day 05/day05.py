from tools import intcode, loader, parsers

pc = intcode.Intcode(parsers.lines(loader.get()))
print(f'Part 1: {pc.run([1])}')  # 14155342
pc = intcode.Intcode(parsers.lines(loader.get()))
print(f'Part 2: {pc.run([5])}')  # 8684145
