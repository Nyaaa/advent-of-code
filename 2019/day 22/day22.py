from tools import parsers, loader


class Shuffle:
    def __init__(self):
        self.rules = parsers.lines(loader.get())
        self.deck = list(range(10007))

    def part_1(self):
        for rule in self.rules:
            try:
                val = int(rule.split()[-1])
            except ValueError:
                self.deck.reverse()
            else:
                if 'cut' in rule:
                    val = int(rule.split()[-1])
                    self.deck = self.deck[val:] + self.deck[:val]
                elif 'increment' in rule:
                    val = int(rule.split()[-1])
                    _deck = [0] * len(self.deck)
                    for i, j in enumerate(self.deck):
                        _deck[(i * val) % len(_deck)] = j
                    self.deck = _deck
        return self.deck.index(2019)


print(Shuffle().part_1())  # 1879
