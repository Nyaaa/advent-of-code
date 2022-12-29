from tools import parsers
from dataclasses import dataclass


@dataclass
class Node:
    row: int
    col: int


class Sensor:
    def __init__(self, node: Node):
        self.node = node
        self.beacon = self

    def calc_distance(self, other: Node):
        return abs(self.node.row - other.row) + abs(self.node.col - other.col)

    @property
    def distance(self):
        return self.calc_distance(self.beacon.node)

    def __repr__(self):
        return f'(({self.node.row}, {self.node.col}), {self.distance})'

    def is_in_range(self, other) -> bool:
        dist = self.calc_distance(other)
        if dist <= self.distance:
            return True
        else:
            return False


def main(data):
    sensors = []
    beacons = []
    in_range = 0

    for line in data:
        newstr = ''.join((ch if ch in '0123456789-' else ' ') for ch in line)
        coord = [int(i) for i in newstr.split()]
        sensor = Sensor(Node(coord[0], coord[1]))
        beacon = Sensor(Node(coord[2], coord[3]))
        sensor.beacon = beacon
        sensors.append(sensor)
        beacons.append(beacon.node)

    min_x = min(sensor.node.row - sensor.distance for sensor in sensors)
    max_x = max(sensor.node.row + sensor.distance for sensor in sensors)

    for sensor in sensors:
        dist = abs(sensor.node.col - y)
        if dist > sensor.distance:
            sensors.remove(sensor)

    for x in range(min_x, max_x):
        check = Node(x, y)
        # print('*' * 20, x)
        for sensor in sensors:
            if sensor.is_in_range(check) and check not in beacons:
                in_range += 1
                break
    return in_range


# test
y = 10
test_result = main(parsers.lines('test15.txt'))
assert test_result == 26

# part 1
y = 2000000
result = main(parsers.lines('input15.txt'))
print(result)  # 5108096
