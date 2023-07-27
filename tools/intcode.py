import logging


class Intcode:
    def __init__(self, data: list):
        self.data = [int(i) for i in data[0].split(',')]
        self.copy = self.data.copy()

    @staticmethod
    def parse_opcode(opcode: int) -> tuple[int, list[int]]:
        string = f'{opcode:05d}'
        return int(string[3:5]), [int(i) for i in string[2::-1]]

    def get_value(self, index: int, mode: int) -> int:
        try:
            return self.copy[self.copy[index]] if mode == 0 else self.copy[index]
        except IndexError:
            pass

    def run(self, input_value: list[int, ...] = None) -> int | list:
        i = 0
        self.copy = self.data.copy()
        if input_value is None:
            input_value = []
        inp = iter(input_value)
        output = 0
        return_output = False
        while i <= len(self.copy):
            opcode, modes = self.parse_opcode(self.copy[i])
            param1 = self.get_value(i + 1, modes[0])
            param2 = self.get_value(i + 2, modes[1])
            match opcode:
                case 1:  # add
                    self.copy[self.copy[i + 3]] = param1 + param2
                    i += 4
                case 2:  # mul
                    self.copy[self.copy[i + 3]] = param1 * param2
                    i += 4
                case 3:  # user input
                    try:
                        val = next(inp)
                    except StopIteration:
                        raise ValueError('Not enough inputs.')
                    if not isinstance(val, int):
                        raise ValueError('Provide an integer input value.')
                    else:
                        self.copy[self.copy[i + 1]] = val
                        i += 2
                case 4:  # print to screen
                    output = param1
                    logging.debug(output)
                    return_output = True
                    i += 2
                case 5 if param1 != 0:  # jump if true
                    i = param2
                case 6 if param1 == 0:  # jump if false
                    i = param2
                case 5 | 6:
                    i += 3
                case 7:  # less than
                    self.copy[self.copy[i + 3]] = 1 if param1 < param2 else 0
                    i += 4
                case 8:  # equals
                    self.copy[self.copy[i + 3]] = 1 if param1 == param2 else 0
                    i += 4
                case 99:  # end of program
                    return output if return_output else self.copy
                case _:
                    raise NotImplementedError(f'Unknown opcode: {opcode}')
        raise IndexError('Pointer value is too high')
