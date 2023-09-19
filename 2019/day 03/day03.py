from tools import parsers, loader


TEST0 = """R8,U5,L5,D3
U7,R6,D4,L4"""
TEST1 = """R75,D30,R83,U83,L12,D49,R71,U7,L72
U62,R66,U55,R34,D71,R55,D58,R83"""
TEST2 = """R98,U47,R26,D63,R33,U87,L62,D20,R33,U53,R51
U98,R91,D20,R16,D67,R40,U7,R15,U6,R7"""


class PCB:
    directions = {'R': 1j, 'L': -1j, 'U': 1, 'D': -1}

    def __init__(self, data: list[str]) -> None:
        self.wires = []
        for line in data:
            new_wire = list()
            current_point = 0j
            for turn in line.split(','):
                for _ in range(int(turn[1:])):
                    current_point += self.directions[turn[:1]]
                    new_wire.append(current_point)
            self.wires.append(new_wire)

    def start(self, part2: bool) -> int:
        """
        >>> print(PCB(parsers.inline_test(TEST1)).start(False))
        159

        >>> print(PCB(parsers.inline_test(TEST2)).start(False))
        135

        >>> print(PCB(parsers.inline_test(TEST1)).start(True))
        610

        >>> print(PCB(parsers.inline_test(TEST2)).start(True))
        410"""
        start = 0j
        lowest_distance = float('inf')
        for i in set(self.wires[0]).intersection(self.wires[1]):
            if not part2:
                curr_distance = int(abs(start.real - i.real) + abs(start.imag - i.imag))
            else:
                curr_distance = self.wires[0].index(i) + self.wires[1].index(i) + 2
            if curr_distance < lowest_distance and curr_distance != 0:
                lowest_distance = curr_distance
        return lowest_distance


print(PCB(parsers.lines(loader.get())).start(part2=False))  # 273
print(PCB(parsers.lines(loader.get())).start(part2=True))  # 15622
