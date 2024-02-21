from tools import loader, parsers


class CRT:
    def __init__(self, data: list[str]) -> None:
        self.next_line = iter(data)
        self.image = ''
        self.out = {}
        self.run()

    def run(self) -> None:
        x = 1
        tick = timer = h_pos = value = 0
        cmd = ''

        while True:
            if tick % 40 == 0:
                self.image += '\n'
                h_pos = 0
            elif tick == 20 or (tick + 20) % 40 == 0:
                self.out[tick] = x
            tick += 1
            h_pos += 1

            if timer > 1:
                timer -= 1
            else:
                if cmd == 'addx' and timer == 1:
                    x += value
                try:
                    line = next(self.next_line).split()
                except StopIteration:
                    break

                cmd = line[0]
                value = int(line[1]) if len(line) > 1 else None
                timer = 2 if cmd == 'addx' else 1

            self.image += '█' if x <= h_pos <= x + 2 else '⠀'

    def part_1(self) -> int:
        """test part 1:
        >>> print(CRT(parsers.lines('test10.txt')).part_1())
        13140
        """
        return sum([i * self.out[i] for i in self.out])

    def part_2(self) -> str:
        """test part 1:
        >>> print(CRT(parsers.lines('test10.txt')).part_2())
        ██⠀⠀██⠀⠀██⠀⠀██⠀⠀██⠀⠀██⠀⠀██⠀⠀██⠀⠀██⠀⠀██⠀⠀
        ███⠀⠀⠀███⠀⠀⠀███⠀⠀⠀███⠀⠀⠀███⠀⠀⠀███⠀⠀⠀███⠀
        ████⠀⠀⠀⠀████⠀⠀⠀⠀████⠀⠀⠀⠀████⠀⠀⠀⠀████⠀⠀⠀⠀
        █████⠀⠀⠀⠀⠀█████⠀⠀⠀⠀⠀█████⠀⠀⠀⠀⠀█████⠀⠀⠀⠀⠀
        ██████⠀⠀⠀⠀⠀⠀██████⠀⠀⠀⠀⠀⠀██████⠀⠀⠀⠀⠀⠀████
        ███████⠀⠀⠀⠀⠀⠀⠀███████⠀⠀⠀⠀⠀⠀⠀███████⠀⠀⠀⠀⠀
        """
        return self.image.strip()


print(CRT(parsers.lines(loader.get())).part_1())  # 15680
print(CRT(parsers.lines(loader.get())).part_2())  # ZFBFHGUP
