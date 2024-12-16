from __future__ import annotations

from typing import NamedTuple

import networkx as nx
import numpy as np

from tools import loader, parsers

DIRECTIONS = {1, -1, 1j, -1j}


class Point(NamedTuple):
    location: complex
    direction: complex


def reindeer_maze(data: list[str], part2: bool) -> int:
    """
    >>> print(reindeer_maze(parsers.lines('test.txt'), part2=False))
    7036
    >>> print(reindeer_maze(parsers.lines('test2.txt'), part2=False))
    11048
    >>> print(reindeer_maze(parsers.lines('test.txt'), part2=True))
    45
    >>> print(reindeer_maze(parsers.lines('test2.txt'), part2=True))
    64"""
    arr = np.array([list(i) for i in data], dtype=str)
    maze = nx.DiGraph()
    maze.add_nodes_from((Point(location=complex(*i), direction=d), {'char': char})
                        for d in DIRECTIONS for i, char in np.ndenumerate(arr) if char != '#')
    start = next(point for point, attr in maze.nodes(data=True)
                 if attr['char'] == 'S' and point.direction == 1j)
    stop = next(point for point, attr in maze.nodes(data=True)
                if attr['char'] == 'E' and point.direction == 1j)

    for point in maze.nodes:
        next_points = [
            Point(location=point.location + point.direction, direction=point.direction),
            Point(location=point.location, direction=point.direction * -1j),
            Point(location=point.location, direction=point.direction * 1j),
        ]
        maze.add_edges_from(
            (point, next_point, {'weight': 1 if next_point.direction == point.direction else 1000})
            for next_point in next_points if next_point in maze.nodes)

    maze.add_edges_from((Point(stop.location, direction), stop, {'weight': 0})
                        for direction in DIRECTIONS)

    if not part2:
        score = nx.dijkstra_path_length(maze, source=start, target=stop, weight='weight')
    else:
        paths = nx.all_shortest_paths(maze, source=start, target=stop, weight='weight')
        score = len({point.location for path in paths for point in path})
    return score


print(reindeer_maze(parsers.lines(loader.get()), part2=False))  # 104516
print(reindeer_maze(parsers.lines(loader.get()), part2=True))  # 545
