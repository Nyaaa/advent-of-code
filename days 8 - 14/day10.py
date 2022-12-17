x = 1
tick = 0
out = {}

with open('input10.txt') as f:
    data = f.read().splitlines()


def get_next_command():
    for piece in data:
        yield piece


time = 0
cmd = None
value = None
offset = 20
next_line = get_next_command()
while True:
    tick += 1
    if time > 1:
        time -= 1

    else:
        if cmd == 'addx' and time == 1:
            x += value
        try:
            line = next(next_line)
        except StopIteration:
            break

        line = line.split()
        cmd = line[0]
        try:
            value = int(line[1])
        except IndexError:
            value = None

        if cmd == 'addx':
            time = 2
        else:
            time = 1

    if tick == 20:
        out[tick] = x
        offset = 60
    elif tick == offset:
        out[tick] = x
        offset += 40

    # print(tick, cmd, value, x)

# part 1

result = []
for i in out:
    step, val = i, out[i]
    result.append(step*val)

print(out)
print(sum(result))  # 15680
