import networkx as nx

from tools import loader, parsers


class MemorySpace:
    def __init__(self, data: list[str], size: int) -> None:
        self.data = [tuple(map(int, i.split(','))) for i in data]
        self.start = (0, 0)
        self.end = (size, size)
        self.graph = nx.grid_2d_graph(size + 1, size + 1)

    def part_1(self, map_number: int) -> int:
        """
        >>> print(MemorySpace(parsers.lines('test.txt'), size=6).part_1(12))
        22"""
        self.graph.remove_nodes_from(self.data[:map_number])
        return nx.shortest_path_length(self.graph, self.start, self.end)

    def part_2(self) -> str:
        """
        >>> print(MemorySpace(parsers.lines('test.txt'), size=6).part_2())
        6,1"""
        for byte in self.data:
            self.graph.remove_node(byte)
            if not nx.has_path(self.graph, self.start, self.end):
                return ','.join(map(str, byte))
        raise ValueError


print(MemorySpace(parsers.lines(loader.get()), size=70).part_1(1024))  # 280
print(MemorySpace(parsers.lines(loader.get()), size=70).part_2())  # 28,56
