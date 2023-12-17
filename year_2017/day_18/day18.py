from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue

from tools import loader, parsers


class Duet:
    def __init__(self, data: list[str], prog_id: int) -> None:
        self.mem = defaultdict(int)
        self.mem['p'] = prog_id
        self.commands = data
        self.step = 0
        self.send_queue = Queue()

    def get_value(self, value: str) -> int:
        try:
            return int(value)
        except ValueError:
            return self.mem[value]

    def run(self, receive: Queue[int] | None = None, send: Queue[int] | None = None) -> int:
        step = count = 0
        if send is None:
            send = self.send_queue

        while True:
            match self.commands[step].split():
                case 'snd', x:
                    send.put(self.get_value(x))
                    count += 1
                case 'set', x, y:
                    self.mem[x] = self.get_value(y)
                case 'add', x, y:
                    self.mem[x] += self.get_value(y)
                case 'mul', x, y:
                    self.mem[x] *= self.get_value(y)
                case 'mod', x, y:
                    self.mem[x] %= self.get_value(y)
                case 'rcv', x if receive:
                    try:
                        self.mem[x] = receive.get(timeout=1)
                    except Empty:
                        return count
                case 'rcv', x if self.get_value(x) != 0:
                    return send.queue[-1]
                case 'jgz', x, y if self.get_value(x) > 0:
                    step += self.get_value(y)
                    continue
            step += 1


def part_1(commands: list[str]) -> int:
    """
    >>> print(part_1(parsers.lines('test.txt')))
    4"""
    return Duet(commands, 0).run()


def part_2(commands: list[str]) -> int:
    a = Duet(commands, 0)
    b = Duet(commands, 1)
    with ThreadPoolExecutor() as executor:
        executor.submit(a.run, a.send_queue, b.send_queue)
        future = executor.submit(b.run, b.send_queue, a.send_queue)
    return future.result()


print(part_1(parsers.lines(loader.get())))  # 8600
print(part_2(parsers.lines(loader.get())))  # 7239
