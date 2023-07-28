from unittest import TestCase
from tools.intcode import Intcode


class TestIntcode(TestCase):
    def test_position_mode_add(self):
        actual, _ = Intcode(['1,1,1,4,99,5,6,0,99']).run()
        self.assertEqual(actual[0], 30)

    def test_immediate_mode_add(self):
        actual, _ = Intcode(['1101,100,-1,4,0']).run()
        self.assertEqual(actual[4], 99)

    def test_immediate_mode_mul(self):
        actual, _ = Intcode(['1002,4,3,4,33']).run()
        self.assertEqual(actual[4], 99)

    def test_position_mode_eq(self):
        actual = Intcode(['3,9,8,9,10,9,4,9,99,-1,8']).run([8])
        self.assertEqual(actual, (1, True))

    def test_position_mode_lt(self):
        actual = Intcode(['3,9,7,9,10,9,4,9,99,-1,8']).run([7])
        self.assertEqual(actual, (1, True))

    def test_immediate_mode_eq(self):
        actual = Intcode(['3,3,1108,-1,8,3,4,3,99']).run([8])
        self.assertEqual(actual, (1, True))

    def test_immediate_mode_lt(self):
        actual = Intcode(['3,3,1107,-1,8,3,4,3,99']).run([7])
        self.assertEqual(actual, (1, True))

    def test_position_mode_jump(self):
        actual = Intcode(['3,12,6,12,15,1,13,14,13,4,13,99,-1,0,1,9']).run([0])
        self.assertEqual(actual, (0, True))

    def test_immediate_mode_jump(self):
        actual = Intcode(['3,3,1105,-1,9,1101,0,0,12,4,12,99,1']).run([0])
        self.assertEqual(actual, (0, True))

    def test_large_number(self):
        actual = Intcode(['104,1125899906842624,99']).run([0])
        self.assertEqual(actual, (1125899906842624, True))

    def test_large_number2(self):
        actual = Intcode(['1102,34915192,34915192,7,4,7,99,0']).run([0])
        self.assertEqual(actual, (1219070632396864, True))

    def test_relative_mode(self):
        actual = Intcode(['109, 1, 203, 2, 204, 2, 99']).run([999])
        self.assertEqual(actual, (999, True))
