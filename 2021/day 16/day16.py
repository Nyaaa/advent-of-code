import math

from tools import loader, parsers

HEX_TO_BIN = {
    '0': '0000',
    '1': '0001',
    '2': '0010',
    '3': '0011',
    '4': '0100',
    '5': '0101',
    '6': '0110',
    '7': '0111',
    '8': '1000',
    '9': '1001',
    'A': '1010',
    'B': '1011',
    'C': '1100',
    'D': '1101',
    'E': '1110',
    'F': '1111'
}


class Decoder:
    def __init__(self, data: str) -> None:
        self.encoded: str = ''.join(HEX_TO_BIN[i] for i in data)
        self.packets: list[int] = []

    def decode_packet(self, cur: int = 0) -> tuple[int, int]:
        pak_version = int(self.encoded[cur:cur + 3], 2)
        pak_type = int(self.encoded[cur + 3:cur + 6], 2)
        cur += 6  # version + type
        if pak_type == 4:
            cur, value = self.decode_literal(cur)
        else:
            cur, values = self.decode_operator(cur)
            value = self.operations(pak_type, *values)
        self.packets.append(pak_version)
        return cur, value

    @staticmethod
    def operations(pak_type: int, *args: int) -> int | None:
        match pak_type:
            case 0: result = sum(args)
            case 1: result = math.prod(args)
            case 2: result = min(args)
            case 3: result = max(args)
            case 5: result = 1 if args[0] > args[1] else 0
            case 6: result = 1 if args[0] < args[1] else 0
            case 7: result = 1 if args[0] == args[1] else 0
            case _: result = None  # should never be the case
        return result

    def decode_operator(self, cur: int) -> tuple[int, list[int]]:
        """ type 0:
        >>> print(Decoder('38006F45291200').part_1())
        9

        type 1:
        >>> print(Decoder('EE00D40C823060').part_1())
        14"""
        values: list[int] = []
        if self.encoded[cur] == '0':
            cur += 16  # prefix + 15
            chunk_end = cur + int(self.encoded[cur - 15:cur], 2)
            while cur < chunk_end:
                cur, value = self.decode_packet(cur)
                values.append(value)
        else:
            cur += 12  # prefix + 11
            packet_amount = int(self.encoded[cur - 11:cur], 2)
            for _ in range(packet_amount):
                cur, value = self.decode_packet(cur)
                values.append(value)
        return cur, values

    def decode_literal(self, cur: int) -> tuple[int, int]:
        binary = ''
        while True:
            prefix = self.encoded[cur]
            cur += 5  # prefix + 4
            binary += self.encoded[cur - 4:cur]
            if prefix == '0':
                return cur, int(binary, 2)

    def part_1(self) -> int:
        """test part 1:
        >>> print(Decoder('8A004A801A8002F478').part_1())
        16

        >>> print(Decoder('620080001611562C8802118E34').part_1())
        12

        >>> print(Decoder('C0015000016115A2E0802F182340').part_1())
        23

        >>> print(Decoder('A0016C880162017C3686B18A3D4780').part_1())
        31
        """
        self.decode_packet()
        return sum(self.packets)

    def part_2(self) -> int:
        """test part 2:
        >>> print(Decoder('C200B40A82').part_2())
        3

        >>> print(Decoder('04005AC33890').part_2())
        54

        >>> print(Decoder('880086C3E88112').part_2())
        7

        >>> print(Decoder('CE00C43D881120').part_2())
        9

        >>> print(Decoder('D8005AC2A8F0').part_2())
        1

        >>> print(Decoder('F600BC2D8F').part_2())
        0

        >>> print(Decoder('9C005AC2F8F0').part_2())
        0

        >>> print(Decoder('9C0141080250320F1802104A08').part_2())
        1
        """
        return self.decode_packet()[1]


print(Decoder(parsers.string(loader.get())).part_1())  # 1002
print(Decoder(parsers.string(loader.get())).part_2())  # 1673210814091
