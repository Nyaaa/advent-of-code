from tools import loader, parsers

TEST = """939
7,13,x,x,59,x,31,19
"""


class Timetable:
    def __init__(self, data: list[str]) -> None:
        self.data = {int(bus): i for i, bus in enumerate(data[1].split(',')) if bus != 'x'}
        self.timestamp = int(data[0])

    def get_next_departure(self, bus: int) -> int:
        """
        >>> print(Timetable(parsers.inline_test(TEST)).get_next_departure(7))
        945
        """
        departure = bus * (self.timestamp // bus)
        if departure < self.timestamp:
            departure += bus
        return departure

    def part_1(self) -> int:
        """
        >>> print(Timetable(parsers.inline_test(TEST)).part_1())
        295
        """
        shortest_wait = float('inf')
        best_bus = 0
        for bus in self.data:
            departure = self.get_next_departure(bus)
            wait = departure - self.timestamp
            if wait <= shortest_wait:
                shortest_wait = wait
                best_bus = bus
        return shortest_wait * best_bus

    def part_2(self) -> int:
        """
        >>> print(Timetable(parsers.inline_test(TEST)).part_2())
        1068781
        """
        time, step = 0, 1
        for bus, index in self.data.items():
            while (time + index) % bus != 0:
                time += step
            step *= bus
        return time


print(Timetable(parsers.lines(loader.get())).part_1())  # 104
print(Timetable(parsers.lines(loader.get())).part_2())  # 842186186521918
