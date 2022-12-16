test = ['R 4', 'U 4', 'L 3', 'D 1', 'R 4', 'D 1', 'L 5', 'R 2']
data = test
# with open('input09.txt') as f:
#     data = f.read().splitlines()



class Board:
    def __init__(self, length: int):
        self.head = [(0, 0)]  # x, y
        self.tail = [(0, 0)]

        self.rope = []
        for i in range(length):
            i = [0, 0]  # x, y
            self.rope.append(i)

    def move_head(self, side):
        current_pos = self.head[-1]
        x, y = current_pos[0], current_pos[1]
        if side == 'R': x += 1
        elif side == 'L': x -= 1
        elif side == 'U': y += 1
        elif side == 'D': y -= 1
        new_pos = (x, y)
        self.head.append(new_pos)
        print(f'head: {current_pos} -> {new_pos}')
        self.move_tail(new_pos)

    def move_tail(self, head_pos):
        hx, hy = head_pos[0], head_pos[1]
        tail_pos = self.tail[-1]
        tx, ty = tail_pos[0], tail_pos[1]
        moved = None

        # horizontal movement
        if hx - tx >= 2: tx += 1; moved='x'
        elif hx - tx <= -2: tx -= 1; moved='x'
        # vertical movement
        if hy - ty >= 2: ty += 1; moved='y'
        elif hy - ty <= -2: ty -= 1; moved='y'
        # diagonal movement
        if moved == 'x' and ty != hy: ty = hy
        elif moved == 'y' and tx != hx: tx = hx

        new_pos = (tx, ty)
        self.tail.append(new_pos)
        if moved: print(f'tail: {tail_pos} -> {new_pos}')


rope = Board(2)
for row in data:
    side, steps = row.split()
    steps = int(steps)
    print('moving', side, steps)
    for i in range(steps):
        rope.move_head(side)

# part 1
print(len(set(rope.tail)))  # 6209

# part 2
print(rope.rope)
