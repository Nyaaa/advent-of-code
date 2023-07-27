import logging


class Intcode:
    def __init__(self, data: list):
        self.data = [int(i) for i in data[0].split(',')]
        self.step = 0

    @staticmethod
    def parse_opcode(opcode: int) -> tuple[int, list[int]]:
        string = f'{opcode:05d}'
        return int(string[3:5]), [int(i) for i in string[2::-1]]

    def get_value(self, index: int, mode: int) -> int:
        try:
            return self.data[self.data[index]] if mode == 0 else self.data[index]
        except IndexError:
            pass

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
        return_output = False
        while self.step <= len(self.data):
            opcode, modes = self.parse_opcode(self.data[self.step])
            param1 = self.get_value(self.step + 1, modes[0])
            param2 = self.get_value(self.step + 2, modes[1])
            match opcode:
                case 1:  # add
                    self.data[self.data[self.step + 3]] = param1 + param2
                    self.step += 4
                case 2:  # mul
                    self.data[self.data[self.step + 3]] = param1 * param2
                    self.step += 4
                case 3:  # user input
                    try:
                        val = next(inp)
                    except StopIteration:
                        # self.data = self.copy
                        # halt at current machine state
                        # run again with new inputs to continue
                        return output, False
                    if not isinstance(val, int):
                        raise ValueError('Provide an integer input value.')
                    else:
                        self.data[self.data[self.step + 1]] = val
                        self.step += 2
                case 4:  # print to screen
                    output = param1
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
                    self.data[self.data[self.step + 3]] = 1 if param1 < param2 else 0
                    self.step += 4
                case 8:  # equals
                    self.data[self.data[self.step + 3]] = 1 if param1 == param2 else 0
                    self.step += 4
                case 99:  # end of program
                    return output if return_output else self.data, True
                case _:
                    raise NotImplementedError(f'Unknown opcode: {opcode}')
        raise IndexError('Pointer value is too high')
