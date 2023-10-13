from collections import defaultdict

from more_itertools import sliced

from tools import intcode, loader, parsers


class Network:
    def __init__(self) -> None:
        program = parsers.lines(loader.get())
        self.net = {i: intcode.Intcode(program) for i in range(50)}
        self.queue = defaultdict(list)
        for i, pc in self.net.items():
            pc.run([i])
        self.idle = {i: False for i in self.net}

    def start(self, part2: bool) -> int:
        result = 0
        while True:
            if all(self.idle.values()):
                q = self.queue.pop(255)
                self.queue[0] = q
                if result == q[1]:
                    return result
                result = q[1]
            for i, pc in self.net.items():
                packets = self.queue.pop(i, [-1])
                output = pc.run(packets)
                if not output:
                    self.idle[i] = True
                    continue
                self.idle[i] = False
                for addr, x, y in sliced(output, 3):
                    self.queue[addr].extend([x, y])
                    if addr == 255:
                        if not part2:
                            return y
                        self.queue[255] = [x, y]


print(Network().start(part2=False))  # 19473
print(Network().start(part2=True))  # 12475
