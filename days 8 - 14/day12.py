from tools import parsers
import string
import networkx as nx

letters = list(string.ascii_lowercase)
numbers = list(range(26))
weights = dict(zip(letters, numbers))
weights['S'] = 0
weights['E'] = 25
test = """Sabqponm
abcryxxl
accszExk
acctuvwj
abdefghi
"""
# data = parsers.inline_test(test)
data = parsers.lines('input12.txt')


class Graph:
    def __init__(self, array: list, start: str):
        self.grid = []
        self.start_coord = []
        self.end_coord = None
        self.start_letter = start
        self.end_letter = 'E'
        self.digraph = nx.DiGraph()

        for row in range(len(array)):
            line = array[row]
            new_row = []
            for pos in range(len(line)):
                letter = line[pos]
                if letter == self.start_letter:
                    self.start_coord.append((row, pos))
                elif letter == self.end_letter:
                    self.end_coord = (row, pos)
                new_row.append(letter)
            self.grid.append(new_row)
        self.build_graph()

    def build_graph(self):
        for row in range(len(self.grid)):
            for pos in range(len(self.grid[row])):
                if row > 0:
                    up = self.grid[row - 1][pos]
                    self.add_edges(up, row, pos, (row - 1, pos))

                if row < (len(self.grid) - 1):
                    down = self.grid[row + 1][pos]
                    self.add_edges(down, row, pos, (row + 1, pos))

                if pos > 0:
                    left = self.grid[row][pos - 1]
                    self.add_edges(left, row, pos, (row, pos - 1))

                if pos < (len(self.grid[row]) - 1):
                    right = self.grid[row][pos + 1]
                    self.add_edges(right, row, pos, (row, pos + 1))

    def add_edges(self, char, row, pos, v):
        letter = self.grid[row][pos]
        weight = weights[letter]
        if weights[char] <= weight + 1:
            self.digraph.add_edge((row, pos), v)

    def solve(self):
        distance = float('inf')
        for point in self.start_coord:
            try:
                attempt = nx.shortest_path_length(self.digraph, source=point, target=self.end_coord)
            except nx.exception.NetworkXNoPath:
                continue
            if attempt <= distance:
                distance = attempt
        return distance


# part 1
g = Graph(data, 'S')
print(g.solve())  # 449

# part 2
g = Graph(data, 'a')
print(g.solve())  # 443

