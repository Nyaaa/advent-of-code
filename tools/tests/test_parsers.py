from unittest import TestCase
from tools import parsers

test_input = 'testcase1.txt'
inline = """move 1 from 2 to 1
move 3 from 1 to 3
move 2 from 2 to 1
move 1 from 1 to 2"""


class Test(TestCase):
    def test_lines(self):
        actual = parsers.lines(test_input)
        expected = ['1000', '2000', '3000', '', '4000', '', '5000', '6000', '', '7000', '8000', '9000', '', '10000']
        self.assertEqual(actual, expected)

    def test_blocks(self):
        actual = parsers.blocks(test_input)
        expected = [['1000', '2000', '3000'], ['4000'], ['5000', '6000'], ['7000', '8000', '9000'], ['10000']]
        self.assertEqual(actual, expected)

    def test_inline_test(self):
        actual = parsers.inline_test(inline)
        expected = ['move 1 from 2 to 1', 'move 3 from 1 to 3', 'move 2 from 2 to 1', 'move 1 from 1 to 2']
        self.assertEqual(actual, expected)

    def test_generator(self):
        gen = parsers.generator(parsers.lines(test_input))
        actual = list(gen)
        expected = ['1000', '2000', '3000', '', '4000', '', '5000', '6000', '', '7000', '8000', '9000', '', '10000']
        self.assertEqual(actual, expected)
