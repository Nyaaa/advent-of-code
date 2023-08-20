from tools import parsers, loader


class Recipe:
    def __init__(self, target: str):
        self.target = target
        self.board = [3, 7]
        self.elves = [0, 1]  # board index

    def step(self):
        a = self.board[self.elves[0]]
        b = self.board[self.elves[1]]
        self.board.extend(int(i) for i in str(a + b))
        for i, j in enumerate(self.elves):
            self.elves[i] = (j + int(self.board[j]) + 1) % len(self.board)

    def part_1(self):
        """
        >>> print(Recipe('2018').part_1())
        5941429882"""
        target = int(self.target)
        while len(self.board) <= target + 10:
            self.step()
        return ''.join(str(i) for i in self.board[target:target + 10])

    def part_2(self):
        """
        >>> print(Recipe('59414').part_2())
        2018"""
        target = [int(i) for i in self.target]
        while True:
            self.step()
            if self.board[-len(target):] == target or self.board[-len(target) - 1:-1] == target:
                break
        return ''.join(str(i) for i in self.board).index(self.target)


print(Recipe(parsers.string(loader.get())).part_1())  # 3610281143
print(Recipe(parsers.string(loader.get())).part_2())  # 20211326
