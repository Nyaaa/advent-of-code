from tools import loader, parsers

test = ['R 4', 'U 4', 'L 3', 'D 1', 'R 4', 'D 1', 'L 5', 'R 2']
test1 = ['R 5', 'U 8', 'L 8', 'D 3', 'R 17', 'D 10', 'L 25', 'U 20']
d = parsers.lines(loader.get())


class Rope:
    def __init__(self, length: int) -> None:
        """
        :param length: amount of rope segments
        """
        self.length = length
        self.segments = [[(0, 0)] for _ in range(self.length)]  # x, y

    def move_head(self, side: str) -> None:
        current_pos = self.segments[0][-1]
        x, y = current_pos[0], current_pos[1]
        if side == 'R': x += 1
        elif side == 'L': x -= 1
        elif side == 'U': y += 1
        elif side == 'D': y -= 1
        self.segments[0].append((x, y))

        for index in range(1, self.length):
            self.move_tail(index)

    def move_tail(self, index: int) -> None:
        head_pos = self.segments[index-1][-1]
        tail_pos = self.segments[index][-1]
        tx, ty = tail_pos[0], tail_pos[1]
        diff = (head_pos[0] - tx, head_pos[1] - ty)

        match diff:
            # horizontal movement
            case (2, 0):
                tx += 1
            case (-2, 0):
                tx -= 1
            # vertical movement
            case (0, 2):
                ty += 1
            case (0, -2):
                ty -= 1
            # diagonal movement
            case (1, 2) | (2, 1) | (2, 2):
                tx += 1
                ty += 1
            case (1, -2) | (2, -1) | (2, -2):
                tx += 1
                ty -= 1
            case (-1, 2) | (-2, 1) | (-2, 2):
                tx -= 1
                ty += 1
            case (-1, -2) | (-2, -1) | (-2, -2):
                tx -= 1
                ty -= 1

        new_pos = (tx, ty)
        self.segments[index].append(new_pos)

    def start(self, data: list[str]) -> int:
        """test part 1:
        >>> print(Rope(2).start(test))
        13

        test part 2:
        >>> print(Rope(10).start(test))
        1
        >>> print(Rope(10).start(test1))
        36
        """
        for row in data:
            side, steps = row.split()
            steps = int(steps)
            for _ in range(steps):
                self.move_head(side)
        return len(set(self.segments[-1]))


# part 1
print(Rope(2).start(d))  # 6209

# part 2
print(Rope(10).start(d))  # 2460
