import re
from dataclasses import dataclass, field
from tools import parsers, loader


@dataclass
class Chunk:
    mask: str
    commands: list = field(default_factory=list)

    def write(self):
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


class Bitmap:
    def __init__(self, data: list):
        self.program = []
        for line in data:
            if 'mask' in line:
                self.program.append(Chunk(mask=line.split(' = ')[1]))
            else:
                addr, val = re.findall(r'\d+', line)
                self.program[-1].commands.append((int(addr), int(val)))

    def part_1(self):
        """
        >>> print(Bitmap(parsers.lines('test.txt')).part_1())
        165"""
        result = {}
        for chunk in self.program:
            result.update(chunk.write())
        return sum(result.values())


print(Bitmap(parsers.lines(loader.get())).part_1())  # 5875750429995
