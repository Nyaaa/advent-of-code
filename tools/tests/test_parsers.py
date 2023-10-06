from pathlib import Path
from unittest import TestCase

from tools import parsers

test_input = Path(__file__).resolve().parent.joinpath('testcase1.txt')
inline = """move 1 from 2 to 1
move 3 from 1 to 3
move 2 from 2 to 1
move 1 from 1 to 2"""


class Test(TestCase):
    def test_lines(self) -> None:
        actual = parsers.lines(test_input)
        expected = ['1000', '2000', '3000', '',
                    '4000', '',
                    '5000', '6000', '',
                    '7000', '8000', '9000', '',
                    '10000']
        self.assertEqual(actual, expected)

    def test_blocks(self) -> None:
        actual = parsers.blocks(test_input)
        expected = [['1000', '2000', '3000'],
                    ['4000'],
                    ['5000', '6000'],
                    ['7000', '8000', '9000'],
                    ['10000']]
        self.assertEqual(actual, expected)

    def test_inline_test(self) -> None:
        actual = parsers.inline_test(inline)
        expected = ['move 1 from 2 to 1',
                    'move 3 from 1 to 3',
                    'move 2 from 2 to 1',
                    'move 1 from 1 to 2']
        self.assertEqual(actual, expected)
