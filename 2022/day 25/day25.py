from tools import parsers, loader

SNAFU_INT = {'1': 1, '2': 2, '0': 0, '-': -1, '=': -2}
INT_SNAFU = {1: '1', 2: '2', 0: '0', 3: '=', 4: '-'}


class Snafu:
    def __init__(self, data: list[str]):
        self.numbers = data

    @staticmethod
    def snafu_to_decimal(snafu: str) -> int:
        """
        >>> print(Snafu([]).snafu_to_decimal('2=-01'))
        976"""
        return sum([SNAFU_INT[val] * (5 ** (abs(pos) - 1)) for pos, val in enumerate(snafu, start=-len(snafu))])

    @staticmethod
    def decimal_to_snafu(dec: int) -> str:
        """
        >>> print(Snafu([]).decimal_to_snafu(976))
        2=-01"""
        snafu = ''
        while dec > 0:
            snafu += INT_SNAFU[dec % 5]
            dec = round(dec / 5)
        return snafu[::-1]

    def part_1(self):
        """
        >>> print(Snafu(parsers.lines('test.txt')).part_1())
        2=-1=0"""
        decimals = [self.snafu_to_decimal(num) for num in self.numbers]
        return self.decimal_to_snafu(sum(decimals))


print(Snafu(parsers.lines(loader.get())).part_1())  # 2=10---0===-1--01-20


