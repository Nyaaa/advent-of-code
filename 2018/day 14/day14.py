from tools import parsers, loader


class Recipe:
    def __init__(self, target: list[str]):
        self.target = target[0]
        self.board = [3, 7]
        self.elves = [0, 1]  # board index

    def create_recipes(self):
        a = self.board[self.elves[0]]
        b = self.board[self.elves[1]]
        self.board.extend(int(i) for i in str(a + b))

    def get_next_recipe(self):
        for i, j in enumerate(self.elves):
            self.elves[i] = (j + int(self.board[j]) + 1) % len(self.board)

    def part_1(self):
        """
        >>> print(Recipe(['2018']).part_1())
        5941429882"""
        target = int(self.target)
        while len(self.board) <= target + 10:
            self.create_recipes()
            self.get_next_recipe()
        return ''.join(str(i) for i in self.board[target:target + 10])


print(Recipe(parsers.lines(loader.get())).part_1())  # 3610281143
