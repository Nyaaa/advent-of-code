import re
from itertools import combinations

from tools import intcode, loader, parsers


class Game:
    def __init__(self) -> None:
        self.pc = intcode.Intcode(parsers.lines(loader.get()))

    def execute(self, command: str) -> list | int:
        return self.pc.run([ord(i) for i in command + '\n'])

    def manage_inv(self, _items: list | tuple, operation: str) -> None:
        for n in _items:
            self.execute(f'{operation} {n}')

    def collect_items(self) -> list[str]:
        for i in parsers.lines('path.txt'):
            self.execute(i)

        items = self.execute('inv')
        text = ''.join(chr(i) for i in items)
        items = text.replace('- ', '').split('\n')
        return [i for i in items if i and i[0].islower()]

    def find_combination(self) -> int:
        items = self.collect_items()
        self.manage_inv(items, 'drop')

        while True:
            for amount in range(1, 9):
                for to_test in combinations(items, amount):
                    self.manage_inv(to_test, 'take')
                    if self.execute('west') == 10:
                        out = ''.join(chr(i) for i in self.pc.logged_output)
                        out = re.findall(r'\d+', out)
                        return int(out[0])
                    self.manage_inv(to_test, 'drop')


print(Game().find_combination())  # 1073815584
