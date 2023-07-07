from collections import deque

from tools import parsers, loader


class Combat:
    def __init__(self, data: list):
        self.player1 = deque([int(i) for i in data[0][1:]])
        self.player2 = deque([int(i) for i in data[1][1:]])

    def play(self):
        winner = None
        while self.player1 and self.player2:
            card1 = self.player1.popleft()
            card2 = self.player2.popleft()
            winner = self.player1 if card1 > card2 else self.player2
            winner.append(card1 if winner == self.player1 else card2)
            winner.append(card2 if winner == self.player1 else card1)
        return winner

    def part_1(self):
        """"
        >>> print((Combat(parsers.blocks('test.txt')).part_1()))
        306"""
        winner = self.play()
        score = 0
        for i, val in enumerate(reversed(winner)):
            score += val * (i + 1)
        return score


print((Combat(parsers.blocks(loader.get())).part_1()))  # 35370
