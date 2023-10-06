"""A module for loading input files from submodule"""
import inspect
import os.path
import re
from pathlib import Path


def get() -> str:
    """Gets the input of the same day as the caller module.
    Module and input files must be named identically, e.g.:
    day01.py, day01.txt

    :return: Input file path
    """

    caller_filename_full = inspect.stack()[1].filename
    day = _get_day_number(caller_filename_full)
    year = os.path.splitext(os.path.basename(Path(caller_filename_full).parent.parent))[0]
    root = Path(__file__).parent.parent
    path = os.path.join(root, 'aoc-inputs', year, f'day{day}.txt')
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def _get_day_number(path: str) -> str:
    """Find all two-digit occurrences in path."""

    candidates = re.findall(r'(?<!\d)\d{2}(?!\d)', path)
    if not candidates:
        raise FileNotFoundError('Could not determine day number.')
    return candidates[-1]
