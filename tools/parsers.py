"""A collection of parsers for processing input files"""


def lines(file: str, strip: bool = True) -> list[str]:
    """Parses input file line by line.

    :param file: Text input
    :param strip: Remove leading/trailing whitespace
    :return: List of lines in strings
    """

    with open(file) as f:
        return [line.strip() if strip else line for line in f.readlines()]


def string(file: str) -> str:
    """Parses input file consisting of a single string.

    :param file: Text input
    :return: Single string
    """

    with open(file) as f:
        return f.read().strip()


def blocks(file: str) -> list[list[str]]:
    """Parses input file in blocks, separated by empty line.

    :param file: Text input
    :return: List of blocks in lists
    """

    with open(file) as f:
        result = []
        for line in f.read().split('\n\n'):
            result.append([i for i in line.split('\n') if i != ''])
        return result


def inline_test(string: str) -> list[str]:
    """Parses input string to match the expected output of line parser.

    :param string: String
    :return: List[str]
    """

    return [line.strip() for line in string.splitlines()]
