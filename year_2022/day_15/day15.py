import re
from typing import NamedTuple

from shapely import clip_by_rect, union_all
from shapely.geometry import LineString, Polygon, mapping

from tools import loader, parsers


class Sensor(NamedTuple):
    row: int
    col: int
    distance: int

    def get_coverage(self) -> Polygon:
        return Polygon([(self.row - self.distance, self.col),
                        (self.row, self.col - self.distance),
                        (self.row + self.distance, self.col),
                        (self.row, self.col + self.distance)])


class Main:
    def __init__(self, data: list[str]) -> None:
        self.sensors = []
        self.area = []
        self.in_range = 0
        self.min_x = 0
        self.max_x = 0

        for line in data:
            coord = list(map(int, re.findall('-?\\d+', line)))
            distance = abs(coord[0] - coord[2]) + abs(coord[1] - coord[3])
            sensor = Sensor(coord[0], coord[1], distance)
            self.sensors.append(sensor)
            self.area.append(sensor.get_coverage())

        self.min_x = min(sensor.row - sensor.distance for sensor in self.sensors)
        self.max_x = max(sensor.row + sensor.distance for sensor in self.sensors)

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
        FIXME: test fails
        >>> Main(parsers.lines('test15.txt')).part_2(20)
        56000011"""
        merge = union_all(self.area)
        clip = clip_by_rect(merge, 0, 0, lim, lim).interiors[0]
        point = mapping(clip.centroid)
        x, y = point.get('coordinates')
        return int(x) * 4000000 + int(y)


print(Main(parsers.lines(loader.get())).part_1(2000000))  # 5108096
print(Main(parsers.lines(loader.get())).part_2(4000000))  # 10553942650264
