from tools import parsers, loader


TEST = """5764801
17807724"""


class Decoder:
    def __init__(self, data: list):
        self.door, self.card = [int(i) for i in data]

    def decode(self):
        """
        >>> print(Decoder(parsers.inline_test(TEST)).decode())
        14897079"""
        loop_size = 0
        value = 1
        while value != min(self.door, self.card):
            value = (value * 7) % 20201227
            loop_size += 1
        return pow(max(self.door, self.card), loop_size, 20201227)


print(Decoder(parsers.lines(loader.get())).decode())  # 1890859
