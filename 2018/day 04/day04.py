import re
from collections import Counter, defaultdict

from tools import loader, parsers


class Guards:
    def __init__(self, data: list[str]) -> None:
        self.timetable = defaultdict(list)
        guard_id = None
        sleep_start = None
        for line in sorted(data):
            time, action, _id = re.findall(r'\[(.*)] (\w+) #?(\d+)?', line)[0]
            time = int(time[-2:])
            if _id:
                guard_id = int(_id)
            if 'falls' in action:
                sleep_start = time
            if 'wakes' in action:
                for x in range(sleep_start, time):
                    self.timetable[guard_id].append(x)

    def part_1(self) -> int:
        """
        >>> print(Guards(parsers.lines('test.txt')).part_1())
        240"""
        guard, times = max(self.timetable.items(), key=lambda i: len(i[1]))
        return guard * Counter(times).most_common(1)[0][0]

    def part_2(self) -> int:
        """
        >>> print(Guards(parsers.lines('test.txt')).part_2())
        4455"""
        longest_sleep = 0
        out = 0
        for guard, times in self.timetable.items():
            minute, sleep = Counter(times).most_common(1)[0]
            if sleep > longest_sleep:
                longest_sleep = sleep
                out = guard * minute
        return out


print(Guards(parsers.lines(loader.get())).part_1())  # 35184
print(Guards(parsers.lines(loader.get())).part_2())  # 37886
