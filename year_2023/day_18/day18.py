import re

from shapely.geometry import Polygon

from tools import loader, parsers


def lagoon(data: list[str], part2: bool) -> int:
    """
    >>> print(lagoon(parsers.lines('test.txt'), part2=False))
    62
    >>> print(lagoon(parsers.lines('test.txt'), part2=True))
    952408144115"""
    directions = {'R': (1, 0), 'L': (-1, 0), 'U': (0, 1), 'D': (0, -1)}
    encoded_directions = {'0': 'R', '1': 'D', '2': 'L', '3': 'U'}
    location = (0, 0)
    corners = [location]
    for line in data:
        direction, distance, colour = re.findall(r'\w+', line)
        if part2:
            distance = int(colour[:-1], 16)
            direction = encoded_directions[colour[-1]]
        side = directions[direction]
        location = (location[0] + side[0] * int(distance), location[1] + side[1] * int(distance))
        corners.append(location)
    polygon = Polygon(corners).buffer(0.5, join_style='mitre')
    return int(polygon.area)


print(lagoon(parsers.lines(loader.get()), part2=False))  # 50603
print(lagoon(parsers.lines(loader.get()), part2=True))  # 96556251590677
