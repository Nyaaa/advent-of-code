from __future__ import annotations

import re
from dataclasses import dataclass, field
from heapq import heappop, heappush

from tools import loader, parsers


@dataclass
class Computer:
    a: int
    b: int
    c: int
    pointer: int = 0
    output: list[int] = field(default_factory=list)

    def run(self, program: list[int]) -> list[int]:
        opcodes = {
            0: self.adv,
            1: self.bxl,
            2: self.bst,
            3: self.jnz,
            4: self.bcx,
            5: self.out,
            6: self.bdv,
            7: self.cdv,
        }
        while True:
            try:
                opcode = program[self.pointer]
            except IndexError:
                return self.output
            op = program[self.pointer + 1]
            jump = opcodes[opcode](op)
            if not jump:
                self.pointer += 2

    def get_combo(self, operand: int) -> int:
        match operand:
            case 4: operand = self.a
            case 5: operand = self.b
            case 6: operand = self.c
        return operand

    def div(self, a: int, b: int) -> int:
        operand = self.get_combo(b)
        return a // (2 ** operand)

    def adv(self, operand: int) -> None:
        self.a = self.div(self.a, operand)

    def bxl(self, operand: int) -> None:
        self.b ^= operand

    def bst(self, operand: int) -> None:
        self.b = self.get_combo(operand) % 8

    def jnz(self, operand: int) -> bool:
        if self.a != 0:
            self.pointer = operand
            return True
        return False

    def bcx(self, _: int) -> None:
        self.b ^= self.c

    def out(self, operand: int) -> None:
        self.output.append(self.get_combo(operand) % 8)

    def bdv(self, operand: int) -> None:
        self.b = self.div(self.a, operand)

    def cdv(self, operand: int) -> None:
        self.c = self.div(self.a, operand)


def part1(data: list[list[str]]) -> str:
    """
    >>> print(part1(parsers.blocks('test.txt')))
    4,6,3,5,6,3,5,2,1,0"""
    registers_, program_ = data
    registers = [line.split()[-1] for line in registers_]
    program = list(map(int, re.findall(r'\d+', program_[0])))
    pc = Computer(*map(int, registers))
    return ','.join(map(str, pc.run(program)))


def part2(data: list[list[str]]) -> int:
    """
    >>> print(part2(parsers.blocks('test2.txt')))
    117440"""
    program_ = data[1]
    program = list(map(int, re.findall(r'\d+', program_[0])))

    potential_a = []
    i = 1
    heappush(potential_a, (i, 0))
    while i < len(program):
        i, a_ = heappop(potential_a)
        for j in range(8):
            a = 8 * a_ + j
            pc = Computer(a, 0, 0)
            output = pc.run(program)
            if output == program:
                return a
            if output == program[-i:]:
                heappush(potential_a, (i + 1, a))
    raise ValueError


print(part1(parsers.blocks(loader.get())))  # 1,3,7,4,6,4,2,3,5
print(part2(parsers.blocks(loader.get())))  # 202367025818154
