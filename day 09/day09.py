test = ['R 4', 'U 4', 'L 3', 'D 1', 'R 4', 'D 1', 'L 5', 'R 2']
test1 = ['R 5', 'U 8', 'L 8', 'D 3', 'R 17', 'D 10', 'L 25', 'U 20']

with open('input09.txt') as f:
    data = f.read().splitlines()


class Rope:
    def __init__(self, length: int):
        self.length = length
        self.segments = []
        for i in range(self.length):
            i = [(0, 0)]  # x, y
            self.segments.append(i)

    def move_head(self, side):
        current_pos = self.segments[0][-1]
        x, y = current_pos[0], current_pos[1]
        if side == 'R': x += 1
        elif side == 'L': x -= 1
        elif side == 'U': y += 1
        elif side == 'D': y -= 1
        new_pos = (x, y)
        self.segments[0].append(new_pos)

        for index in range(1, self.length):
            self.move_tail(index)

    def move_tail(self, index: int):
        head_pos = self.segments[index-1][-1]
        tail_pos = self.segments[index][-1]
        tx, ty = tail_pos[0], tail_pos[1]
        diff = (head_pos[0] - tx, head_pos[1] - ty)

        match diff:
            # horizontal movement
            case (2, 0): tx += 1
            case (-2, 0): tx -= 1
            # vertical movement
            case (0, 2): ty += 1
            case (0, -2): ty -= 1
            # diagonal movement
            case (1, 2) | (2, 1) | (2, 2): tx += 1; ty += 1
            case (1, -2) | (2, -1) | (2, -2): tx += 1; ty -= 1
            case (-1, 2) | (-2, 1) | (-2, 2): tx -= 1; ty += 1
            case (-1, -2) | (-2, -1) | (-2, -2): tx -= 1; ty -= 1

        new_pos = (tx, ty)
        self.segments[index].append(new_pos)


def start(length: int) -> int:
    """length: amount of rope segments"""
    r = Rope(length)
    for row in data:
        side, steps = row.split()
        steps = int(steps)
        for _ in range(steps):
            r.move_head(side)
    return len(set(r.segments[-1]))


# part 1
print(start(2))  # 6209

# part 2
print(start(10))  # 2460
