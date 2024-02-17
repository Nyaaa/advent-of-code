from tools import loader, parsers


class Rope:
    def __init__(self, length: int) -> None:
        """
        :param length: amount of rope segments
        """
        self.length = length
        self.segments = [[0j] for _ in range(self.length)]
        self.movements = {'R': 1, 'L': -1, 'U': 1j, 'D': -1j}

    def move_head(self, side: str) -> None:
        current_pos = self.segments[0][-1]
        self.segments[0].append(current_pos + self.movements[side])
        for index in range(1, self.length):
            self.move_tail(index)

    def move_tail(self, index: int) -> None:
        head_pos = self.segments[index - 1][-1]
        tail_pos = self.segments[index][-1]

        match head_pos - tail_pos:
            # horizontal movement
            case 2:
                move = 1
            case -2:
                move = -1
            # vertical movement
            case 2j:
                move = 1j
            case -2j:
                move = -1j
            # diagonal movement
            case 1+2j | 2+1j | 2+2j:
                move = 1+1j
            case 1-2j | 2-1j | 2-2j:
                move = 1-1j
            case -1+2j | -2+1j | -2+2j:
                move = -1+1j
            case -1-2j | -2-1j | -2-2j:
                move = -1-1j
            case _:
                move = 0j

        self.segments[index].append(tail_pos + move)

    def start(self, data: list[str]) -> int:
        """
        >>> print(Rope(2).start(parsers.lines('test.txt')))
        13
        >>> print(Rope(10).start(parsers.lines('test.txt')))
        1
        >>> print(Rope(10).start(parsers.lines('test2.txt')))
        36
        """
        for row in data:
            side, steps = row.split()
            for _ in range(int(steps)):
                self.move_head(side)
        return len(set(self.segments[-1]))


print(Rope(2).start(parsers.lines(loader.get())))  # 6209
print(Rope(10).start(parsers.lines(loader.get())))  # 2460
