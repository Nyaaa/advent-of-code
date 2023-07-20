from collections import deque

from tools import parsers, loader


class Combat:
    def __init__(self, data: list):
        self.player1 = deque([int(i) for i in data[0][1:]])
        self.player2 = deque([int(i) for i in data[1][1:]])

    def play(self) -> deque:
        winner = None
        while self.player1 and self.player2:
            card1 = self.player1.popleft()
            card2 = self.player2.popleft()
            winner = self.player1 if card1 > card2 else self.player2
            winner.append(card1 if winner == self.player1 else card2)
            winner.append(card2 if winner == self.player1 else card1)
        return winner

    def play_recursive(self, player1: deque, player2: deque) -> deque:
        seen = set()

        while player1 and player2:
            self.player1 = player1
            self.player2 = player2
            state = (tuple(player1), tuple(player2))
            if state in seen:
                return self.player1
            seen.add(state)

            card1 = player1.popleft()
            card2 = player2.popleft()
            if card1 <= len(player1) and card2 <= len(player2):
                winner = self.play_recursive(deque(list(player1)[:card1]),
                                             deque(list(player2)[:card2]))
            else:
                winner = self.player1 if card1 > card2 else self.player2

            if winner == self.player1:
                player1.extend([card1, card2])
            else:
                player2.extend([card2, card1])
        return self.player1 or self.player2

    def start(self, part2: bool) -> int:
        """"
        >>> print((Combat(parsers.blocks('test.txt')).start(False)))
        306

        >>> print((Combat(parsers.blocks('test.txt')).start(True)))
        291"""
        winner = self.play() if not part2 else self.play_recursive(self.player1, self.player2)
        score = 0
        for i, val in enumerate(reversed(winner)):
            score += val * (i + 1)
        return score


print((Combat(parsers.blocks(loader.get())).start(False)))  # 35370
print((Combat(parsers.blocks(loader.get())).start(True)))  # 36246
