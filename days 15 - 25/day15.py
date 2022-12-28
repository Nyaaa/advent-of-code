from tools import parsers

# data = parsers.lines('input15.txt')
data = parsers.lines('test15.txt')

sensors = set()
beacons = set()


def is_in_range(row: int, col: int) -> bool:
    for sensor in sensors:
        dis = abs(row - sensor[0]) + abs(col - sensor[1])
        print(sensor, dis)
        if dis <= sensor[2]:
            print(True)
            return True
    # print(False)
    return False



for line in data:
    newstr = ''.join((ch if ch in '0123456789-' else ' ') for ch in line)
    coord = [int(i) for i in newstr.split()]
    s_row, s_col = coord[0], coord[1]
    b_row, b_col = coord[2], coord[3]
    distance = abs(s_row - b_row) + abs(s_col - b_col)
    sensors.add((s_row, s_col, distance))
    beacons.add((b_row, b_col))


min_x = min(sublist[0] for sublist in sensors | beacons)
max_x = max(sublist[0] for sublist in sensors | beacons)

print(sensors)
print(beacons)

# part 1
y = 10
in_range = 0

for x in range(min_x, max_x):
    if is_in_range(x, y):
        in_range += 1
print(in_range)  # fails

