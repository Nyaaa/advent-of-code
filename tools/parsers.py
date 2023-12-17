"""A collection of parsers for processing input files"""
import re
from pathlib import Path


def lines(file: str | Path, strip: bool = True) -> list[str]:
    """Parses input file line by line.

    Args:
        file: Text input
        strip: Remove leading/trailing whitespace

    Returns:
        List of lines in strings
    """

    with Path.open(file) as f:
        return [line.strip() if strip else line for line in f.readlines()]


def string(file: str | Path) -> str:
    """Parses input file consisting of a single string.

    Args:
        file: Text input

    Returns:
        Single string
    """

    with Path.open(file) as f:
        return f.read().strip()


def blocks(file: str | Path) -> list[list[str]]:
    """Parses input file in blocks, separated by empty line.

    Args:
        file: Text input

    Returns:
        List of blocks in lists
    """

    with Path.open(file) as f:
        return [[i for i in line.split('\n') if i]
                for line in re.split(r'\n\n+', f.read())]


def inline_test(text: str) -> list[str]:
    """Parses input string to match the expected output of line parser.

    Args:
        text: String

    Returns:
        List of strings
    """

    return [line.strip() for line in text.splitlines()]
