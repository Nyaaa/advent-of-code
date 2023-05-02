from collections import deque
from dataclasses import dataclass
from typing import Optional
import logging
from tools import parsers, loader

logging.basicConfig(level=logging.DEBUG)
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
        self.children = []

    def decode_packet(self, packet: str):
        pak_version = int(packet[:3], 2)
        pak_type = int(packet[3:6], 2)
        if pak_type == 4:
            decoded, remainder = self.decode_literal(packet[6:])
            bb = Packet(version=pak_version, type=pak_type, value=decoded)
            self.packets.append(bb)
            self.children.append(decoded)
            if remainder:
                self.decode_packet(remainder)
            return self.children
        else:
            parent = Packet(version=pak_version, type=pak_type)
            child = self.decode_operator(packet)
            parent.children = child
            logging.debug(parent)

            self.packets.append(parent)
            return parent

    def decode_operator(self, packet: str):
        """ type 0:
        >>> print(Decoder(parsers.inline_test('38006F45291200')).part_1())
        9

        type 1:
        >>> print(Decoder(parsers.inline_test('EE00D40C823060')).part_1())
        14"""
        length_type = packet[6]
        if length_type == '0':
            child = self.decode_packet(packet[7 + 15:])
        else:
            child = self.decode_packet(packet[7 + 11:])

        if isinstance(child, list):
            child = child.copy()
            self.children.clear()
        return child

    @staticmethod
    def decode_literal(encoded: str) -> tuple[int, str | None]:
        """ Represents a single number.
        >>> print(Decoder(parsers.inline_test('D2FE28')).decode_packet('110100101111111000101000')[0])
        2021"""
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
        logging.debug(self.packets)
        return sum([i.version for i in self.packets])


# print(Decoder(parsers.lines(loader.get())).decode())  # 1002
print(Decoder(parsers.inline_test('04005AC33890')).part_1())

