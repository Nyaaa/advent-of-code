from tools import parsers, loader, intcode

pc = intcode.Intcode(parsers.lines(loader.get()))
print(pc.run([1]))  # 2494485073
pc = intcode.Intcode(parsers.lines(loader.get()))
print(pc.run([2]))  # 44997
