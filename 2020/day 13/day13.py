from tools import parsers, loader

TEST = """939
7,13,x,x,59,x,31,19
"""


class Timetable:
    def __init__(self, data: list):
        self.data = [int(i) for i in data[1].split(',') if i != 'x']
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

    def part_1(self):
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


print(Timetable(parsers.lines(loader.get())).part_1())  # 104

