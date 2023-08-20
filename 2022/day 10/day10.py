from tools import parsers, loader

d = parsers.lines(loader.get())
t = parsers.lines('test10.txt')


class CRT:
    def __init__(self, data):
        self.next_line = iter(data)
        self.image = ['', '', '', '', '', '']
        self.out = {}
        self.run()

    def run(self, x=1, tick=0, timer=0, offset=20, v_line=0, row=-1, cmd=None, value=None):
        while True:
            if tick % 40 == 0:
                row += 1
                v_line = 0
            elif tick == 20:
                self.out[tick] = x
                offset = 60
            elif tick == offset:
                self.out[tick] = x
                offset += 40
            tick += 1
            v_line += 1

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

            sprite = (x, x + 1, x + 2)
            self.image[row] += '█' if v_line in sprite else '⠀'

    def part_1(self):
        """test part 1:
        >>> print(CRT(t).part_1())
        13140
        """
        return sum([i * self.out[i] for i in self.out])

    def part_2(self):
        """test part 1:
        >>> print(CRT(t).part_2())
        ██⠀⠀██⠀⠀██⠀⠀██⠀⠀██⠀⠀██⠀⠀██⠀⠀██⠀⠀██⠀⠀██⠀⠀
        ███⠀⠀⠀███⠀⠀⠀███⠀⠀⠀███⠀⠀⠀███⠀⠀⠀███⠀⠀⠀███⠀
        ████⠀⠀⠀⠀████⠀⠀⠀⠀████⠀⠀⠀⠀████⠀⠀⠀⠀████⠀⠀⠀⠀
        █████⠀⠀⠀⠀⠀█████⠀⠀⠀⠀⠀█████⠀⠀⠀⠀⠀█████⠀⠀⠀⠀⠀
        ██████⠀⠀⠀⠀⠀⠀██████⠀⠀⠀⠀⠀⠀██████⠀⠀⠀⠀⠀⠀████
        ███████⠀⠀⠀⠀⠀⠀⠀███████⠀⠀⠀⠀⠀⠀⠀███████⠀⠀⠀⠀⠀
        """
        result = ''
        for row in self.image:
            if result:
                result += '\n'
            for i in row:
                result += i

        return result


print(CRT(d).part_1())  # 15680
print(CRT(d).part_2())  # ZFBFHGUP
