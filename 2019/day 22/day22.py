from tools import parsers, loader


class Shuffle:
    def __init__(self):
        self.rules = parsers.lines(loader.get())

    def part_1(self):
        deck = list(range(10007))
        for rule in self.rules:
            try:
                val = int(rule.split()[-1])
            except ValueError:
                deck.reverse()
            else:
                if 'cut' in rule:
                    val = int(rule.split()[-1])
                    deck = deck[val:] + deck[:val]
                elif 'increment' in rule:
                    val = int(rule.split()[-1])
                    _deck = [0] * len(deck)
                    for i, j in enumerate(deck):
                        _deck[(i * val) % len(_deck)] = j
                    deck = _deck
        return deck.index(2019)

    def part_2(self):
        cards = 119315717514047
        times = 101741582076661
        index = 2020
        a, b = 1, 0
        for rule in self.rules:
            try:
                val = int(rule.split()[-1])
            except ValueError:
                a = -a % cards
                b = (cards - 1 - b) % cards
            else:
                if 'cut' in rule:
                    b -= val % cards
                elif 'increment' in rule:
                    a *= val % cards
                    b *= val % cards
        r = (b * pow(1 - a, cards - 2, cards)) % cards
        return ((index - r) * pow(a, times * (cards - 2), cards) + r) % cards


print(Shuffle().part_1())  # 1879
print(Shuffle().part_2())  # 73729306030290
