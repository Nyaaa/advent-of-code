"""A module for loading input files from submodule"""
import inspect
import re
from pathlib import Path


def get() -> Path:
    """Gets the input of the same day as the caller module.

    :return: Input file path
    """

    caller_filename_full = inspect.stack()[1].filename
    day = _get_day_number(caller_filename_full)
    year = re.findall(r'\d{4}', caller_filename_full)[-1]
    root = Path(__file__).parent.parent
    path = root / 'aoc-inputs' / year / f'day{day}.txt'
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _get_day_number(path: str) -> str:
    """Find all two-digit occurrences in path."""

    candidates = re.findall(r'(?<!\d)\d{2}(?!\d)', path)
    if not candidates:
        raise FileNotFoundError('Could not determine day number.')
    return candidates[-1]
