import logging
from collections import defaultdict


class Intcode:
    def __init__(self, data: list[str]):
        data = [int(i) for i in data[0].split(',')]
        self.data: defaultdict[int, int] = defaultdict(int)
        self.data.update({k: v for k, v in enumerate(data)})
        self.step = 0
        self.relative_base = 0

    @staticmethod
    def parse_opcode(opcode: int) -> tuple[int, list[int]]:
        string = f'{opcode:05d}'
        return int(string[3:5]), [int(i) for i in string[2::-1]]

    def get_value(self, index: int, mode: int) -> int:
        match mode:
            case 0:  # position mode
                return self.data[self.data[index]]
            case 1:  # immediate mode
                return self.data[index]
            case 2:  # relative mode
                return self.data[self.data[index] + self.relative_base]

    def get_write_index(self, index: int, mode: int) -> int:
        """Write locations can never be given in immediate mode as per Day 5 rule.
        Before Day 9 they are in position mode only.
        After Day 9 they can be in relative mode."""
        match mode:
            case 2:
                return self.data[index] + self.relative_base
            case _:
                return self.data[index]

    def run(self, input_value: list[int, ...] = None) -> tuple[int | list, bool]:
        """
        Args:
            input_value: A list of program input values.

        Returns:
            (result, bool) where bool is whether the program properly terminated.
        """
        if input_value is None:
            input_value = []
        inp = iter(input_value)
        output = 0
        full_output = []
        return_output = False
        while self.step <= len(self.data):
            opcode, modes = self.parse_opcode(self.data[self.step])
            param1 = self.get_value(self.step + 1, modes[0])
            param2 = self.get_value(self.step + 2, modes[1])
            param3 = self.get_write_index(self.step + 3, modes[2])
            match opcode:
                case 1:  # add
                    self.data[param3] = param1 + param2
                    self.step += 4
                case 2:  # mul
                    self.data[param3] = param1 * param2
                    self.step += 4
                case 3:  # user input
                    try:
                        val = int(next(inp))
                    except StopIteration:
                        # halt at current machine state
                        # run again with new inputs to continue
                        return full_output, False
                    if not isinstance(val, int):
                        raise ValueError('Provide an integer input value.')
                    else:
                        out = self.get_write_index(self.step + 1, modes[0])
                        self.data[out] = val
                        self.step += 2
                case 4:  # print to screen
                    output = param1
                    full_output.append(output)
                    logging.debug(output)
                    return_output = True
                    self.step += 2
                case 5 if param1 != 0:  # jump if true
                    self.step = param2
                case 6 if param1 == 0:  # jump if false
                    self.step = param2
                case 5 | 6:
                    self.step += 3
                case 7:  # less than
                    self.data[param3] = 1 if param1 < param2 else 0
                    self.step += 4
                case 8:  # equals
                    self.data[param3] = 1 if param1 == param2 else 0
                    self.step += 4
                case 9:  # modify relative base
                    self.relative_base += param1
                    self.step += 2
                case 99:  # end of program
                    return output if return_output else self.data, True
                case _:
                    raise NotImplementedError(f'Unknown opcode: {opcode}')
        raise IndexError('Pointer value is too high')
