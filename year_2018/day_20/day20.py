import networkx as nx
import regex as re

from tools import loader, parsers


class Maze:
    def __init__(self, data: str) -> None:
        self.regex = data[1:]
        self.G = nx.Graph()
        self.directions = {'N': -1, 'S': 1, 'E': 1j, 'W': -1j}

    def follow_path(self, path: str, prev_pos: complex = 0j) -> None:
        start = prev_pos
        index = 0
        while True:
            next_char = path[index]
            if next_char == '(':
                branch = re.search(r'\((?:[^)(]+|(?R))*+\)', path[index:])
                index += branch.end()
                self.follow_path(branch.group()[1:], prev_pos)
                continue
            if next_char == '|':
                return self.follow_path(path[index + 1:], start)
            if next_char in '$)':
                break
            next_char = path[index]
            curr_pos = prev_pos + self.directions[next_char]
            self.G.add_edge(prev_pos, curr_pos)
            prev_pos = curr_pos
            index += 1

    def start(self) -> tuple[int, int]:
        """
        >>> print(Maze('^ESSWWN(E|NNENN(EESS(WNSE|)SSS|WWWSSSSE(SW|NNNE)))$').start())
        (23, 0)"""
        self.follow_path(self.regex)
        lengths = nx.shortest_path_length(self.G, 0j)
        part1 = max(lengths.values())
        part2 = sum(i >= 1000 for i in lengths.values())
        return part1, part2


print(Maze(parsers.string(loader.get())).start())  # 3991, 8394
