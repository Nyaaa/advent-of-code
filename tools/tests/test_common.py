from unittest import TestCase

import numpy as np

from tools import common

ARRAY = np.array(
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ],
    dtype=int,
)


class Test(TestCase):
    def test_get_adjacent(self) -> None:
        results = {
            (1, 1): [((0, 1), 2), ((1, 0), 4), ((1, 2), 6), ((2, 1), 8)],
            # corners
            (0, 0): [((0, 1), 2), ((1, 0), 4)],
            (0, 2): [((0, 1), 2), ((1, 2), 6)],
            (2, 0): [((1, 0), 4), ((2, 1), 8)],
            (2, 2): [((1, 2), 6), ((2, 1), 8)],
            # sides
            (0, 1): [((0, 0), 1), ((0, 2), 3), ((1, 1), 5)],
            (1, 0): [((0, 0), 1), ((1, 1), 5), ((2, 0), 7)],
            (1, 2): [((0, 2), 3), ((1, 1), 5), ((2, 2), 9)],
            (2, 1): [((1, 1), 5), ((2, 0), 7), ((2, 2), 9)],
        }

        for k, v in results.items():
            actual = list(common.get_adjacent(ARRAY, k))
            expected = v
            self.assertEqual(actual, expected)

    def test_get_adjacent_with_corners(self) -> None:
        results = {
            (1, 1): [((0, 1), 2), ((1, 0), 4), ((1, 2), 6), ((2, 1), 8),
                     ((0, 0), 1), ((0, 2), 3), ((2, 0), 7), ((2, 2), 9)],
            # corners
            (0, 0): [((0, 1), 2), ((1, 0), 4), ((1, 1), 5)],
            (0, 2): [((0, 1), 2), ((1, 2), 6), ((1, 1), 5)],
            (2, 0): [((1, 0), 4), ((2, 1), 8), ((1, 1), 5)],
            (2, 2): [((1, 2), 6), ((2, 1), 8), ((1, 1), 5)],
            # sides
            (0, 1): [((0, 0), 1), ((0, 2), 3), ((1, 1), 5), ((1, 0), 4), ((1, 2), 6)],
            (1, 0): [((0, 0), 1), ((1, 1), 5), ((2, 0), 7), ((0, 1), 2), ((2, 1), 8)],
            (1, 2): [((0, 2), 3), ((1, 1), 5), ((2, 2), 9), ((0, 1), 2), ((2, 1), 8)],
            (2, 1): [((1, 1), 5), ((2, 0), 7), ((2, 2), 9), ((1, 0), 4), ((1, 2), 6)],
        }

        for k, v in results.items():
            actual = list(common.get_adjacent(ARRAY, k, with_corners=True))
            expected = v
            self.assertEqual(actual, expected)

    def test_get_adjacent_with_self(self) -> None:
        results = {
            (1, 1): [((0, 1), 2), ((1, 0), 4), ((1, 2), 6), ((2, 1), 8), ((1, 1), 5)],
        }

        for k, v in results.items():
            actual = list(common.get_adjacent(ARRAY, k, with_self=True))
            expected = v
            self.assertEqual(actual, expected)
