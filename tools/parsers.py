"""A collection of parsers for processing input files"""
from typing import Iterator, List


def lines(file: str) -> List[str]:
    with open(file) as f:
        return [line.strip() for line in f.readlines()]


def blocks(file: str) -> List[List[str]]:
    with open(file) as f:
        result = []
        for line in f.read().split('\n\n'):
            result.append(line.split())
        return result


def inline_test(string: str) -> List[str]:
    return [line.strip() for line in string.splitlines()]


def generator(data: List[str]) -> Iterator[str]:
    for piece in data:
        yield piece
