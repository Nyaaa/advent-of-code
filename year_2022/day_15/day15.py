from dataclasses import dataclass

from shapely import clip_by_rect, union_all
from shapely.geometry import LineString, Polygon, mapping

from tools import loader, parsers


@dataclass
class Node:
    row: int
    col: int

    def __hash__(self) -> int:
        return hash(str(self))


@dataclass
class Sensor:
    node: Node
    distance: int


class Main:
    def __init__(self, data: list[str]) -> None:
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
            distance = abs(sensor.row - beacon.row) + abs(sensor.col - beacon.col)
            sensor = Sensor(sensor, distance)
            self.sensors.append(sensor)
            self.area.append(Polygon([(sensor.node.row - sensor.distance, sensor.node.col),
                                      (sensor.node.row, sensor.node.col - sensor.distance),
                                      (sensor.node.row + sensor.distance, sensor.node.col),
                                      (sensor.node.row, sensor.node.col + sensor.distance)]))

        self.min_x = min(sensor.node.row - sensor.distance for sensor in self.sensors)
        self.max_x = max(sensor.node.row + sensor.distance for sensor in self.sensors)

    def part_1(self, y: int) -> int:
        """Calculate coverage areas usig Manhattan distance,
        convert them to shapely polygons,
        find intersections with row Y.
        >>> Main(parsers.lines('test15.txt')).part_1(10)
        26"""
        row_string = LineString([(self.min_x, y), (self.max_x, y)])
        projection = row_string.intersection(union_all(self.area))
        return int(projection.length)

    def part_2(self, lim: int) -> int:
        """Merge all radii, clip them with a bounding box,
        get center point coordinates
        >>> Main(parsers.lines('test15.txt')).part_2(20)
        56000011"""
        merge = union_all(self.area)
        clip = clip_by_rect(merge, 0, 0, lim, lim).interiors[0]
        point = mapping(clip.centroid)
        x, y = point.get('coordinates')
        return int(x) * 4000000 + int(y)


print(Main(parsers.lines(loader.get())).part_1(2000000))  # 5108096
print(Main(parsers.lines(loader.get())).part_2(4000000))  # 10553942650264
