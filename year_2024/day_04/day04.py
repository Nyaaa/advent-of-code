from collections import deque
from typing import NamedTuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from tools import common, loader, parsers


class Word(NamedTuple):
    letters: list[tuple[int, int]]

    def __len__(self) -> int:
        return len(self.letters)

    def is_valid(self) -> bool:
        diffs = set()
        prev = self.letters[0]
        for row, col in self.letters[1:]:
            diffs.add((row - prev[0], col - prev[1]))
            prev = (row, col)
        return len(diffs) == 1


def part1(data: list[str]) -> int:
    """
    >>> print(part1(parsers.lines('test1.txt')))
    18"""
    word = 'XMAS'
    arr = np.genfromtxt(data, delimiter=1, dtype=str)
    possible_matches = deque(
        Word(letters=[(int(i), int(j))]) for i, j in np.argwhere(arr == 'X'))
    found_words = []
    while possible_matches:
        current_word = possible_matches.popleft()
        if len(current_word) == 4:
            found_words.append(current_word)
            continue
        next_letter = word[len(current_word)]
        current_pos = current_word.letters[-1]
        for new_pos, new_letter in common.get_adjacent(arr, current_pos, with_corners=True):
            if new_letter == next_letter:
                letters = current_word.letters.copy()
                letters.append(new_pos)
                new_word = Word(letters=letters)
                if new_word.is_valid():
                    possible_matches.append(new_word)
    return len(found_words)


def part2(data: list[str]) -> int:
    """
    >>> print(part2(parsers.lines('test2.txt')))
    9"""
    masks = {'MSAMS', 'SMASM', 'SSAMM', 'MMASS'}
    mas_count = 0
    arr = np.genfromtxt(data, delimiter=1, dtype=str)
    for window in sliding_window_view(arr, (3, 3)):
        for w in window:
            if ''.join(w.flatten()[::2]) in masks:
                mas_count += 1
    return mas_count


print(part1(parsers.lines(loader.get())))  # 2397
print(part2(parsers.lines(loader.get())))  # 1824
