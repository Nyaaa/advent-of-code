"""A module for loading input files from submodule"""
import inspect
import os.path
from pathlib import Path


def get():
    """Gets the input of the same day as the caller module.
    Module and input files must be named identically, e.g.:
    day01.py, day01.txt

    :return: Input file path
    """

    caller_filename_full = inspect.stack()[1].filename
    day = os.path.splitext(os.path.basename(caller_filename_full))[0]
    year = os.path.splitext(os.path.basename(Path(caller_filename_full).parent.parent))[0]
    root = Path(__file__).parent.parent
    path = os.path.join(root, 'aoc-inputs', year, f'{day}.txt')
    if os.path.isfile(path):
        return path
    else:
        raise FileNotFoundError(path)
