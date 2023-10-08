from tools import loader, parsers


class Recipe:
    def __init__(self, target: str) -> None:
        self.target = target
        self.board = [3, 7]
        self.elves = [0, 1]  # board index

    def step(self) -> None:
        a = self.board[self.elves[0]]
        b = self.board[self.elves[1]]
        self.board.extend(int(i) for i in str(a + b))
        for i, j in enumerate(self.elves):
            self.elves[i] = (j + int(self.board[j]) + 1) % len(self.board)

    def part_1(self) -> str:
        """
        >>> print(Recipe('2018').part_1())
        5941429882"""
        target = int(self.target)
        while len(self.board) <= target + 10:
            self.step()
        return ''.join(str(i) for i in self.board[target:target + 10])

    def part_2(self) -> int:
        """
        >>> print(Recipe('59414').part_2())
        2018"""
        target = [int(i) for i in self.target]
        while not any(
                target == i for i in (self.board[-len(target):], self.board[-len(target) - 1:-1])
        ):
            self.step()

        return ''.join(str(i) for i in self.board).index(self.target)


print(Recipe(parsers.string(loader.get())).part_1())  # 3610281143
print(Recipe(parsers.string(loader.get())).part_2())  # 20211326
