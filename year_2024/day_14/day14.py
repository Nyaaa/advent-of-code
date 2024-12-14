import re
from collections import defaultdict
from dataclasses import dataclass
from itertools import count, pairwise
from math import prod

import numpy as np

from tools import loader, parsers
from tools.common import Point


@dataclass
class Robot:
    location: Point
    velocity: Point


def count_robots(robots: list[Robot], max_rows: int, max_cols: int) -> int:
    quads = defaultdict(int)
    for robot in robots:
        if robot.location.row < max_rows // 2:
            row = 'top'
        elif robot.location.row > max_rows // 2:
            row = 'bottom'
        else:
            continue
        if robot.location.col < max_cols // 2:
            col = 'left'
        elif robot.location.col > max_cols // 2:
            col = 'right'
        else:
            continue
        quads[f'{row} {col}'] += 1

    return prod(quads.values())


def median_distance(robots: list[Robot]) -> np.floating:
    dist = [a.location.manhattan_distance(b.location) for a, b in pairwise(robots)]
    return np.median(dist)


def predict_robots(data: list[str], max_rows: int, max_cols: int, part2: bool) -> int:
    """
    >>> print(predict_robots(parsers.lines('test.txt'), 7, 11, False))
    12"""
    robots = []
    for line in data:
        vals = list(map(int, re.findall(r'-?\d+', line)))
        loc = Point(col=vals[0], row=vals[1])
        vel = Point(col=vals[2], row=vals[3])
        robots.append(Robot(loc, vel))

    for i in count():
        if not part2 and i == 100:
            break
        if part2 and median_distance(robots) < 40:  # 60+ average
            return i
        for robot in robots:
            loc_ = robot.location + robot.velocity
            new_row = loc_.row % max_rows
            new_col = loc_.col % max_cols
            robot.location = Point(row=new_row, col=new_col)

    return count_robots(robots, max_rows, max_cols)


print(predict_robots(parsers.lines(loader.get()), 103, 101, False))  # 236628054
print(predict_robots(parsers.lines(loader.get()), 103, 101, True))  # 7584
