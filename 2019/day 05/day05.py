from tools import parsers, loader, intcode

pc = intcode.Intcode(parsers.lines(loader.get()))
print('Part 1: enter ID 1')
pc.run()  # 14155342
print('Part 2: enter ID 5')
pc.run()  # 8684145
