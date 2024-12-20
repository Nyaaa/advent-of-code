import networkx as nx
import numpy as np

from tools import loader, parsers


def race(data: list[str], cheat_length: int, min_savings: int = 100) -> int:
    """
    >>> print(race(parsers.lines('test.txt'), cheat_length=2, min_savings=1))
    44
    >>> print(race(parsers.lines('test.txt'), cheat_length=20, min_savings=50))
    285"""
    grid = np.asarray([list(line) for line in data])
    start = tuple(np.argwhere(grid == 'S')[0])
    end = tuple(np.argwhere(grid == 'E')[0])
    graph = nx.grid_2d_graph(*grid.shape)
    graph_no_walls = graph.copy()
    graph.remove_nodes_from(tuple(i) for i in np.argwhere(grid == '#'))
    start_distances = nx.shortest_path_length(graph, start)
    end_distances = nx.shortest_path_length(graph, end)
    shortest_path = start_distances[end]

    result = 0
    for point_a, distance_from_start in start_distances.items():
        cheats = nx.single_source_dijkstra_path_length(
            graph_no_walls, point_a, cutoff=cheat_length)
        for point_b, cheat_distance in cheats.items():
            if point_b not in end_distances:
                continue
            total_distance = distance_from_start + cheat_distance + end_distances[point_b]
            if shortest_path - total_distance >= min_savings:
                result += 1
    return result


print(race(parsers.lines(loader.get()), cheat_length=2))  # 1360
print(race(parsers.lines(loader.get()), cheat_length=20))  # 1005476
