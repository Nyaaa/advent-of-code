import math
from collections import deque
from dataclasses import dataclass
from typing import Optional
from tools import parsers, loader

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


@dataclass
class Packet:
    version: int
    type: int
    value: Optional[int] = None
    children: Optional[list] = None


class Decoder:
    def __init__(self, data):
        self.encoded = ''.join(HEX_TO_BIN[i] for i in data[0])
        self.packets = []
        self.root: Optional[Packet]
        self.parent = deque([])
        self.part2: bool = False

    def get_type(self, packet: str) -> int:
        return int(packet[3:6], 2)

    def decode_packet(self, packet: str):
        pak_version = int(packet[:3], 2)
        pak_type = int(packet[3:6], 2)
        pk = Packet(version=pak_version, type=pak_type, children=[])
        if not self.packets:
            self.root = pk
        self.packets.append(pk)
        if pak_type == 4:
            decoded, remainder = self.decode_literal(packet[6:])
            pk.value = decoded
            self.parent[-1].children.append(decoded)
            if remainder:
                following = self.get_type(remainder)
                if following != 4:
                    self.parent.pop()
                self.decode_packet(remainder)
        else:
            if self.parent:
                self.parent[-1].children.append(pk)
            self.parent.append(pk)
            self.decode_operator(packet)
            child = pk.children
            if child and self.part2:
                pk.value = self.operations(pak_type, *child)

    @staticmethod
    def operations(pak_type: int, *args):
        if isinstance(args[0], Packet):
            args = [i.value for i in args]

        match pak_type:
            case 0: result = sum(args)
            case 1: result = math.prod(args)
            case 2: result = min(args)
            case 3: result = max(args)
            case 5: result = 1 if args[0] > args[1] else 0
            case 6: result = 1 if args[0] < args[1] else 0
            case 7: result = 1 if args[0] == args[1] else 0
        return result

    def decode_operator(self, packet: str):
        """ type 0:
        >>> print(Decoder(parsers.inline_test('38006F45291200')).part_1())
        9

        type 1:
        >>> print(Decoder(parsers.inline_test('EE00D40C823060')).part_1())
        14"""
        length_type = packet[6]
        if length_type == '0':
            self.decode_packet(packet[7 + 15:])
        else:
            self.decode_packet(packet[7 + 11:])

    @staticmethod
    def decode_literal(encoded: str) -> tuple[int, str | None]:
        chunks = deque([encoded[i:i + 5] for i in range(0, len(encoded), 5)])
        binary = ''
        while True:
            ch = chunks.popleft()
            match ch[0], ch[1:]:
                case '1', num if len(num) == 4:
                    binary += num
                case '0', num if len(num) == 4:
                    binary += num
                    remainder = ''.join(chunks)
                    remainder_nonzero = remainder.replace('0', '')
                    return int(binary, 2), remainder if remainder_nonzero else None
                case _:
                    return int(binary, 2), None

    def part_1(self):
        """test part 1:
        >>> print(Decoder(parsers.inline_test('8A004A801A8002F478')).part_1())
        16

        >>> print(Decoder(parsers.inline_test('620080001611562C8802118E34')).part_1())
        12

        >>> print(Decoder(parsers.inline_test('C0015000016115A2E0802F182340')).part_1())
        23

        >>> print(Decoder(parsers.inline_test('A0016C880162017C3686B18A3D4780')).part_1())
        31
        """
        self.decode_packet(self.encoded)
        return sum([i.version for i in self.packets])

    def part_2(self):
        """test part 2:
        >>> print(Decoder(parsers.inline_test('C200B40A82')).part_2())
        3

        >>> print(Decoder(parsers.inline_test('04005AC33890')).part_2())
        54

        >>> print(Decoder(parsers.inline_test('880086C3E88112')).part_2())
        7

        >>> print(Decoder(parsers.inline_test('CE00C43D881120')).part_2())
        9

        >>> print(Decoder(parsers.inline_test('D8005AC2A8F0')).part_2())
        1

        >>> print(Decoder(parsers.inline_test('F600BC2D8F')).part_2())
        0

        >>> print(Decoder(parsers.inline_test('9C005AC2F8F0')).part_2())
        0

        >>> print(Decoder(parsers.inline_test('9C0141080250320F1802104A08')).part_2())
        1
        """
        self.part2 = True
        self.decode_packet(self.encoded)
        return self.root.value


# print(Decoder(parsers.lines(loader.get())).part_1())  # 1002
print(Decoder(parsers.lines(loader.get())).part_2())  #
# print(Decoder(parsers.inline_test('D2318C6318C621')).part_1())

