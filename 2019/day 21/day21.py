from tools import intcode, loader, parsers


class SpringScript:
    def __init__(self) -> None:
        self.pc = intcode.Intcode(parsers.lines(loader.get()))

    def start(self, program: list[str]) -> int:
        return self.pc.run([ord(x) for x in '\n'.join(program) + '\n'])

    def part_1(self) -> int:
        program = [
            'NOT A J',
            'NOT B T',
            'OR T J',
            'NOT C T',
            'OR T J',
            'AND D J',
            'WALK'
        ]
        return self.start(program)

    def part_2(self) -> int:
        program = [
            'NOT C J',
            'AND D J',
            'AND H J',
            'NOT B T',
            'AND D T',
            'OR T J',
            'NOT A T',
            'OR T J',
            'RUN'
        ]
        return self.start(program)


print(SpringScript().part_1())  # 19349722
print(SpringScript().part_2())  # 1141685254
