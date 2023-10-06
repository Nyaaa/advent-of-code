from collections import defaultdict

from tools import loader, parsers


class Graph:
    def __init__(self, data: list[str]) -> None:
        self.nodes = defaultdict(list)
        for line in data:
            name, conn = line.split('-')
            self.nodes[name].append(conn)
            self.nodes[conn].append(name)

    def count_paths(
            self, extra_visit: bool, node: str = 'start', seen: tuple = ('start', )
    ) -> int:
        """
        >>> print(Graph(parsers.lines('test.txt')).count_paths(extra_visit=False))
        10

        >>> print(Graph(parsers.lines('test.txt')).count_paths(extra_visit=True))
        36"""
        if node == 'end':
            return 1
        paths = 0

        for conn in self.nodes[node]:
            if not conn.islower() or conn not in seen:
                paths += self.count_paths(extra_visit, conn, seen + (conn, ))
            elif extra_visit and conn != 'start':
                paths += self.count_paths(False, conn, seen + (conn, ))
        return paths


print(Graph(parsers.lines(loader.get())).count_paths(extra_visit=False))  # 4413
print(Graph(parsers.lines(loader.get())).count_paths(extra_visit=True))  # 118803
