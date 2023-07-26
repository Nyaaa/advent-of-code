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

    def run(self) -> list[int]:
        """
        >>> print(Intcode(['1,1,1,4,99,5,6,0,99']).run())
        [30, 1, 1, 4, 2, 5, 6, 0, 99]

        >>> print(Intcode(['1002,4,3,4,33']).run())
        [1002, 4, 3, 4, 99]

        >>> print(Intcode(['1101,100,-1,4,0']).run())
        [1101, 100, -1, 4, 99]
        """
        i = 0
        self.copy = self.data.copy()
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
                    self.copy[self.copy[i + 1]] = int(input('Enter a value: '))
                    i += 2
                case 4:  # print to screen
                    print(param1)
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
                    return self.copy
                case _:
                    raise NotImplementedError(f'Unknown opcode: {opcode}')
        raise IndexError('Pointer value is too high')
