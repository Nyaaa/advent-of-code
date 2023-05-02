from collections import deque
from typing import NamedTuple, Optional

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
BIN_TO_HEX = dict((int(v), k) for k, v in HEX_TO_BIN.items())
TEST = 'D2FE28'


class Packet(NamedTuple):
    version: int
    type: int
    value: Optional[int]


class Decoder:
    def __init__(self, data):
        self.encoded = ''.join(HEX_TO_BIN[i] for i in data[0])
        self.packets = []

    def decode_packet(self, packet: str):
        pak_version: int = int(BIN_TO_HEX[int(packet[:3])])
        pak_type: int = int(BIN_TO_HEX[int(packet[3:6])])
        length_type = int(packet[6])
        remainder = None
        decoded = None
        if pak_type == 4:
            decoded, remainder = self.decode_literal(packet[6:])
        else:
            if length_type == 0:
                self.decode_op_total_len(packet[7:])
            elif length_type == 1:
                self.decode_op_total_amount(packet[7:])
        if remainder:
            self.decode_packet(remainder)
        self.packets.append(Packet(version=pak_version, type=pak_type, value=decoded))
        return decoded

    def decode_op_total_len(self, encoded: str):
        """
        >>> print(Decoder(parsers.inline_test('38006F45291200')).decode())
        9"""
        length: int = int(encoded[:15], 2)
        return self.decode_packet(encoded[15:])

    def decode_op_total_amount(self, encoded: str):
        """
        >>> print(Decoder(parsers.inline_test('EE00D40C823060')).decode())
        14"""
        amount: int = int(encoded[:11], 2)
        return self.decode_packet(encoded[11:])

    def decode_literal(self, encoded: str) -> tuple[int, str | None]:
        """
        >>> print(Decoder(parsers.inline_test('D2FE28')).decode_packet('110100101111111000101000'))
        2021"""
        chunks = deque([encoded[i:i + 5] for i in range(0, len(encoded), 5)])
        binary = ''
        while True:
            ch = chunks.popleft()
            match (ch[0], ch[1:]):
                case ('1', num) if len(num) == 4:
                    binary += num
                case ('0', num) if len(num) == 4:
                    binary += num
                    remainder = ''.join(chunks)
                    remainder_nonzero = remainder.replace('0', '')
                    return int(binary, 2), remainder if remainder_nonzero else None
                case _:
                    return int(binary, 2), None

    def decode(self):
        """test part 1:
        >>> print(Decoder(parsers.inline_test('8A004A801A8002F478')).decode())
        16

        >>> print(Decoder(parsers.inline_test('620080001611562C8802118E34')).decode())
        12

        >>> print(Decoder(parsers.inline_test('C0015000016115A2E0802F182340')).decode())
        23

        >>> print(Decoder(parsers.inline_test('A0016C880162017C3686B18A3D4780')).decode())
        31
        """
        self.decode_packet(self.encoded)
        # print(self.packets)
        return sum([i.version for i in self.packets])


# print(Decoder(parsers.inline_test(TEST)).decode())
print(Decoder(parsers.lines(loader.get())).decode())  # 1002
