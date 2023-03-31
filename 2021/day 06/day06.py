from tools import parsers, loader

TEST = '3,4,3,1,2'


class Fish:
    def __init__(self, data):
        self.school = {n: 0 for n in reversed(range(9))}
        for i in data[0].split(','):
            self.school[int(i)] += 1

    def start(self, timer: int):
        """
        >>> print(Fish(parsers.inline_test(TEST)).start(80))
        5934

        >>> print(Fish(parsers.inline_test(TEST)).start(256))
        26984457539"""
        time = 0
        while time != timer:
            time += 1
            new_dist = {n: 0 for n in reversed(range(9))}
            for i in self.school:
                if i - 1 < 0:
                    new_dist[6] += self.school[i]
                    new_dist[8] += self.school[i]
                else:
                    new_dist[i - 1] = self.school[i]
            self.school = new_dist
        return sum(self.school.values())


print(Fish(parsers.lines(loader.get())).start(80))  # 391671
print(Fish(parsers.lines(loader.get())).start(256))  # 1754000560399
