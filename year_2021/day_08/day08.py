from tools import loader, parsers

DIGITS = {7: 8, 3: 7, 4: 4, 2: 1}
TEST = 'acedgfb cdfbe gcdfa fbcad dab cefabd cdfgeb eafb cagedb ab | cdfeb fcadb cdfeb cdbaf'


class Display:
    def __init__(self, data: list[str]) -> None:
        self.data = [[a.split(), b.split()] for a, b in (i.split(' | ') for i in data)]

    def part_1(self) -> int:
        """
        >>> print(Display(parsers.lines('test.txt')).part_1())
        26"""
        return sum(sum(1 for i in out if len(i) in DIGITS) for _, out in self.data)

    def part_2(self) -> int:
        """
        >>> print(Display(parsers.lines('test.txt')).part_2())
        61229"""
        total = []
        for inp, out in self.data:
            vals = {}
            while len(vals) < 10:
                for i in inp:
                    i = frozenset(i)
                    num = DIGITS.get(len(i))
                    if num:
                        vals[num] = i
                    try:
                        one = len(i.intersection(vals[1]))
                        four = len(i.intersection(vals[4]))
                    except KeyError:
                        continue

                    match (len(i), one, four):
                        case (5, 2, _): vals[3] = i
                        case (5, _, 2): vals[2] = i
                        case (5, _, _): vals[5] = i
                        case (6, 1, _): vals[6] = i
                        case (6, _, 4): vals[9] = i
                        case (6, _, _): vals[0] = i

            di = {v: k for k, v in vals.items()}
            num = ''
            for i in out:
                i = frozenset(i)
                num += str(di[i])
            total.append(int(num))
        return sum(total)


print(Display(parsers.lines(loader.get())).part_1())  # 365
print(Display(parsers.lines(loader.get())).part_2())  # 975706
