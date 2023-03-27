"""A collection of parsers for processing input files"""
from typing import Iterator, List, Any


def lines(file: str) -> List[str]:
    """Parses input file line by line

    :param file: Text input
    :return: List of lines in strings
    """

    with open(file) as f:
        return [line.strip() for line in f.readlines()]


def blocks(file: str) -> List[List[str]]:
    """Parses input file in blocks, separated by empty line

    :param file: Text input
    :return: List of blocks in lists
    """

    with open(file) as f:
        result = []
        for line in f.read().split('\n\n'):
            result.append([i for i in line.split('\n') if i != ''])
        return result


def inline_test(string: str) -> List[str]:
    """Parses input string to match the expected output of line parser

    :param string: String
    :return: List[str]
    """

    return [line.strip() for line in string.splitlines()]


def generator(data: List[Any]) -> Iterator[str]:
    """Returns a parsed input line by line when called

    :param data: (in)Line parser output
    :return: Generator
    """

    for piece in data:
        yield piece
