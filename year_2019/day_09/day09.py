from tools import intcode, loader, parsers

pc = intcode.Intcode(parsers.lines(loader.get()))
print(pc.run([1]))  # 2494485073
pc = intcode.Intcode(parsers.lines(loader.get()))
print(pc.run([2]))  # 44997
