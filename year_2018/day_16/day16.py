import re
import sys
from inspect import getmembers, isfunction

from tools import loader, parsers


def instr_addr(instr: tuple, register: list) -> list:
    register[instr[3]] = register[instr[1]] + register[instr[2]]
    return register


def instr_addi(instr: tuple, register: list) -> list:
    register[instr[3]] = register[instr[1]] + instr[2]
    return register


def instr_mulr(instr: tuple, register: list) -> list:
    register[instr[3]] = register[instr[1]] * register[instr[2]]
    return register


def instr_muli(instr: tuple, register: list) -> list:
    register[instr[3]] = register[instr[1]] * instr[2]
    return register


def instr_banr(instr: tuple, register: list) -> list:
    register[instr[3]] = register[instr[1]] & register[instr[2]]
    return register


def instr_bani(instr: tuple, register: list) -> list:
    register[instr[3]] = register[instr[1]] & instr[2]
    return register


def instr_borr(instr: tuple, register: list) -> list:
    register[instr[3]] = register[instr[1]] | register[instr[2]]
    return register


def instr_bori(instr: tuple, register: list) -> list:
    register[instr[3]] = register[instr[1]] | instr[2]
    return register


def instr_setr(instr: tuple, register: list) -> list:
    register[instr[3]] = register[instr[1]]
    return register


def instr_seti(instr: tuple, register: list) -> list:
    register[instr[3]] = instr[1]
    return register


def instr_gtir(instr: tuple, register: list) -> list:
    register[instr[3]] = 1 if instr[1] > register[instr[2]] else 0
    return register


def instr_gtri(instr: tuple, register: list) -> list:
    register[instr[3]] = 1 if register[instr[1]] > instr[2] else 0
    return register


def instr_gtrr(instr: tuple, register: list) -> list:
    register[instr[3]] = 1 if register[instr[1]] > register[instr[2]] else 0
    return register


def instr_eqir(instr: tuple, register: list) -> list:
    register[instr[3]] = 1 if instr[1] == register[instr[2]] else 0
    return register


def instr_eqri(instr: tuple, register: list) -> list:
    register[instr[3]] = 1 if register[instr[1]] == instr[2] else 0
    return register


def instr_eqrr(instr: tuple, register: list) -> list:
    register[instr[3]] = 1 if register[instr[1]] == register[instr[2]] else 0
    return register


class Program:
    def __init__(self, data: list[list[str]]) -> None:
        self.program = [tuple(map(int, line.split())) for line in data[-1]]
        self.samples = []
        for sample in data[:-1]:
            self.samples.append(list(map(int, re.findall(r'\d+', i))) for i in sample)
        self.instructions = [m for n, m in getmembers(sys.modules[__name__], isfunction)
                             if n.startswith('instr')]
        self.known_instructions = {}

    def part_1(self) -> int:
        out = 0
        for before, instr, after in self.samples:
            candidates = []
            for operation in self.instructions:
                if operation(instr, before.copy()) == after:
                    candidates.append(operation)
            if len(candidates) >= 3:
                out += 1
        return out

    def run_program(self) -> int:
        register = [0, 0, 0, 0]
        for line in self.program:
            operation = self.known_instructions[line[0]]
            register = operation(line, register)
        return register[0]

    def part_2(self) -> int:
        while self.instructions:
            for before, instr, after in self.samples:
                candidates = []
                for operation in self.instructions:
                    if (operation(instr, before.copy()) == after and
                            instr[0] not in self.known_instructions):
                        candidates.append(operation)
                if len(candidates) == 1:
                    self.known_instructions[instr[0]] = candidates[0]
                    self.instructions.remove(candidates[0])
        return self.run_program()


print(Program(parsers.blocks(loader.get())).part_1())  # 563
print(Program(parsers.blocks(loader.get())).part_2())  # 629