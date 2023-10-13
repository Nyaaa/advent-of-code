from more_itertools import ilen, sliced

from tools import loader, parsers
from year_2019 import intcode


class Game:
    def __init__(self) -> None:
        self.pc = intcode.Intcode(parsers.lines(loader.get()))
        self.ball = None
        self.paddle = None
        self.window = {}

    def compose_window(self, output: list[int]) -> None:
        for col, row, _id in sliced(output, 3):
            point = (row, col)
            self.window[point] = _id
            if _id == 3:
                self.paddle = point
            elif _id == 4:
                self.ball = point

    def part_1(self) -> int:
        self.pc.run()
        self.compose_window(self.pc.logged_output)
        return ilen(i for i in self.window.values() if i == 2)

    def part_2(self) -> int:
        self.pc.data[0] = 2
        result = self.pc.run()
        self.compose_window(result)
        while not self.pc.done:
            if self.ball[1] > self.paddle[1]:
                joystick = 1
            elif self.ball[1] < self.paddle[1]:
                joystick = -1
            else:
                joystick = 0
            self.pc.run([joystick])
            result = self.pc.logged_output
            self.compose_window(result)
        return self.window[0, -1]


print(Game().part_1())  # 324
print(Game().part_2())  # 15957
