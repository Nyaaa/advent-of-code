import re
from dataclasses import dataclass, field
from itertools import product

from tools import loader, parsers


@dataclass
class Chunk:
    mask: str
    commands: list = field(default_factory=list)

    def write(self) -> dict[int, int]:
        output = {}
        for addr, val in self.commands:
            bin_val = format(val, 'b').zfill(36)
            out = ''
            for i, v in enumerate(self.mask):
                match v:
                    case '1': out += '1'
                    case '0': out += '0'
                    case _: out += bin_val[i]
            output[addr] = int(out, 2)
        return output

    def write_v2(self) -> dict[int, int]:
        output = {}
        for addr, val in self.commands:
            bin_val = format(addr, 'b').zfill(36)
            out = ''
            for i, v in enumerate(self.mask):
                match v:
                    case '1': out += '1'
                    case '0': out += bin_val[i]
                    case 'X': out += 'X'
            for a in (out.replace('X', '{}').format(*p)
                      for p in product('01', repeat=out.count('X'))):
                output[int(a, 2)] = val
        return output


class Bitmap:
    def __init__(self, data: list) -> None:
        self.program = []
        for line in data:
            if 'mask' in line:
                self.program.append(Chunk(mask=line.split(' = ')[1]))
            else:
                addr, val = re.findall(r'\d+', line)
                self.program[-1].commands.append((int(addr), int(val)))

    def count(self, part2: bool) -> int:
        """
        >>> print(Bitmap(parsers.lines('test.txt')).count(part2=False))
        165

        >>> print(Bitmap(parsers.lines('test2.txt')).count(part2=True))
        208"""
        result = {}
        for chunk in self.program:
            result.update(chunk.write() if not part2 else chunk.write_v2())
        return sum(result.values())


print(Bitmap(parsers.lines(loader.get())).count(part2=False))  # 5875750429995
print(Bitmap(parsers.lines(loader.get())).count(part2=True))  # 5272149590143
