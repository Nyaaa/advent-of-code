import os
from unittest import TestCase

from tools import loader


class Test(TestCase):
    def test_get_day_number(self) -> None:
        path = os.path.join('2021', 'day 01', 'day01', '1', '111', 'test')
        actual = loader._get_day_number(path)
        expected = '01'
        self.assertEqual(actual, expected)

    def test_no_day_number(self) -> None:
        path = os.path.join('2021', 'day 1', 'day1', '1', '111', 'test')
        with self.assertRaises(FileNotFoundError):
            loader._get_day_number(path)
