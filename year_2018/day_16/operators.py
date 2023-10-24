def addr(instr: tuple, register: list) -> list:
    register[instr[3]] = register[instr[1]] + register[instr[2]]
    return register


def addi(instr: tuple, register: list) -> list:
    register[instr[3]] = register[instr[1]] + instr[2]
    return register


def mulr(instr: tuple, register: list) -> list:
    register[instr[3]] = register[instr[1]] * register[instr[2]]
    return register


def muli(instr: tuple, register: list) -> list:
    register[instr[3]] = register[instr[1]] * instr[2]
    return register


def banr(instr: tuple, register: list) -> list:
    register[instr[3]] = register[instr[1]] & register[instr[2]]
    return register


def bani(instr: tuple, register: list) -> list:
    register[instr[3]] = register[instr[1]] & instr[2]
    return register


def borr(instr: tuple, register: list) -> list:
    register[instr[3]] = register[instr[1]] | register[instr[2]]
    return register


def bori(instr: tuple, register: list) -> list:
    register[instr[3]] = register[instr[1]] | instr[2]
    return register


def setr(instr: tuple, register: list) -> list:
    register[instr[3]] = register[instr[1]]
    return register


def seti(instr: tuple, register: list) -> list:
    register[instr[3]] = instr[1]
    return register


def gtir(instr: tuple, register: list) -> list:
    register[instr[3]] = 1 if instr[1] > register[instr[2]] else 0
    return register


def gtri(instr: tuple, register: list) -> list:
    register[instr[3]] = 1 if register[instr[1]] > instr[2] else 0
    return register


def gtrr(instr: tuple, register: list) -> list:
    register[instr[3]] = 1 if register[instr[1]] > register[instr[2]] else 0
    return register


def eqir(instr: tuple, register: list) -> list:
    register[instr[3]] = 1 if instr[1] == register[instr[2]] else 0
    return register


def eqri(instr: tuple, register: list) -> list:
    register[instr[3]] = 1 if register[instr[1]] == instr[2] else 0
    return register


def eqrr(instr: tuple, register: list) -> list:
    register[instr[3]] = 1 if register[instr[1]] == register[instr[2]] else 0
    return register
