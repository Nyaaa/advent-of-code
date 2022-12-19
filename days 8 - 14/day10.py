from tools import parsers
import time


x = 1
tick = 0
out = {}
timer = 0
cmd = None
value = None
offset = 20
data = parsers.lines('input10.txt')
next_line = parsers.generator(data)
v_line = 0
row = -1
image = ['', '', '', '', '', '']


while True:
    if tick % 40 == 0:
        row += 1
        v_line = 0
    tick += 1
    v_line += 1
    if timer > 1:
        timer -= 1

    else:
        if cmd == 'addx' and timer == 1:
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
            timer = 2
        else:
            timer = 1

    if tick == 20:
        out[tick] = x
        offset = 60
    elif tick == offset:
        out[tick] = x
        offset += 40

    sprite = (x, x+1, x+2)
    if v_line in sprite:
        image[row] += '█'
    else:
        image[row] += '⠀'

# part 1

result = []
for i in out:
    step, val = i, out[i]
    result.append(step*val)
print(sum(result))  # 15680

# part 2
# for row in image:
#     print(row)  # ZFBFHGUP

for row in image:
    vrow = ''
    for i in row:
        vrow += i
        print(f'\r{vrow}', end='')
        time.sleep(0.08)
    print('\r')
