from tools import parsers
from dataclasses import dataclass
from shapely import union_all, clip_by_rect
from shapely.geometry import mapping, Polygon, LineString


@dataclass
class Node:
    row: int
    col: int

    def __hash__(self):
        return hash(str(self))


@dataclass
class Sensor:
    node: Node
    distance: int


class Main:
    def __init__(self, data):
        self.sensors = []
        self.area = []
        self.in_range = 0
        self.min_x = 0
        self.max_x = 0

        for line in data:
            newstr = ''.join((ch if ch in '0123456789-' else ' ') for ch in line)
            coord = [int(i) for i in newstr.split()]
            sensor = Node(coord[0], coord[1])
            beacon = Node(coord[2], coord[3])
            distance = abs(sensor.row - beacon.row) + abs(sensor.col - beacon.col)  # Manhattan distance
            sensor = Sensor(sensor, distance)
            self.sensors.append(sensor)
            self.area.append(Polygon([(sensor.node.row - sensor.distance, sensor.node.col),
                                      (sensor.node.row, sensor.node.col - sensor.distance),
                                      (sensor.node.row + sensor.distance, sensor.node.col),
                                      (sensor.node.row, sensor.node.col + sensor.distance)]))

        self.min_x = min(sensor.node.row - sensor.distance for sensor in self.sensors)
        self.max_x = max(sensor.node.row + sensor.distance for sensor in self.sensors)

    def part_1(self, y: int) -> int:
        row_string = LineString([(self.min_x, y), (self.max_x, y)])
        projection = row_string.intersection(union_all(self.area))
        return int(projection.length)

    def part_2(self, lim: int) -> int:
        merge = union_all(self.area)
        clip = clip_by_rect(merge, 0, 0, lim, lim).interiors[0]
        point = mapping(clip.centroid)
        x, y = point.get('coordinates')
        return int(x) * 4000000 + int(y)


def part_1(data, y):
    """Calculate coverage areas usig Manhattan distance,
    convert them to shapely polygons,
    find intersections with row Y.
    >>> part_1(parsers.lines('test15.txt'), 10)
    26"""
    return Main(data).part_1(y)


def part_2(data, y):
    """Merge all radii, clip them with a bounding box,
    get center point coordinates
    >>> part_2(parsers.lines('test15.txt'), 20)
    56000011"""
    return Main(data).part_2(y)


print(part_1(parsers.lines('input15.txt'), 2000000))  # 5108096
print(part_2(parsers.lines('input15.txt'), 4000000))  # 10553942650264
