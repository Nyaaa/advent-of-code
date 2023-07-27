from unittest import TestCase
from tools.intcode import Intcode


class TestIntcode(TestCase):
    def test_position_mode_add(self):
        actual = Intcode(['1,1,1,4,99,5,6,0,99']).run()
        self.assertEqual(actual, [30, 1, 1, 4, 2, 5, 6, 0, 99])

    def test_immediate_mode_add(self):
        actual = Intcode(['1101,100,-1,4,0']).run()
        self.assertEqual(actual, [1101, 100, -1, 4, 99])

    def test_immediate_mode_mul(self):
        actual = Intcode(['1002,4,3,4,33']).run()
        self.assertEqual(actual, [1002, 4, 3, 4, 99])

    def test_position_mode_eq(self):
        actual = Intcode(['3,9,8,9,10,9,4,9,99,-1,8']).run([8])
        self.assertEqual(actual, 1)

    def test_position_mode_lt(self):
        actual = Intcode(['3,9,7,9,10,9,4,9,99,-1,8']).run([7])
        self.assertEqual(actual, 1)

    def test_immediate_mode_eq(self):
        actual = Intcode(['3,3,1108,-1,8,3,4,3,99']).run([8])
        self.assertEqual(actual, 1)

    def test_immediate_mode_lt(self):
        actual = Intcode(['3,3,1107,-1,8,3,4,3,99']).run([7])
        self.assertEqual(actual, 1)

    def test_position_mode_jump(self):
        actual = Intcode(['3,12,6,12,15,1,13,14,13,4,13,99,-1,0,1,9']).run([0])
        self.assertEqual(actual, 0)

    def test_immediate_mode_jump(self):
        actual = Intcode(['3,3,1105,-1,9,1101,0,0,12,4,12,99,1']).run([0])
        self.assertEqual(actual, 0)

