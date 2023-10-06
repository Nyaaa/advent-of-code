"""A collection of common functions for AoC."""
from __future__ import annotations

from collections.abc import Generator
from typing import Any, NamedTuple

import numpy as np
from numba import njit
from numpy.typing import NDArray


@njit()
def get_adjacent(
        array: NDArray,
        position: tuple[int, int],
        with_corners: bool = False,
        with_self: bool = False) -> Generator[tuple[tuple[int, int], Any]]:
    """
    Get adjacent cells in NumPy 2d array. Compiled with Numba.
    Args:
        array (NDArray): NumPy array
        position (tuple): cell (row, col) position
        with_corners (bool): include corner cells
        with_self (bool): include center cell

    Returns:
        (adjacent_row, adjacent_col), adjacent_value
    """

    adjacent = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    if with_corners:
        adjacent += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    if with_self:
        adjacent += [(0, 0)]

    max_rows, max_cols = array.shape

    for i_row, i_col in adjacent:
        adj_row = position[0] + i_row
        adj_col = position[1] + i_col

        if (0 <= adj_row < max_rows) and (0 <= adj_col < max_cols):
            yield (adj_row, adj_col), array[adj_row, adj_col]


def trim_array(array: NDArray) -> NDArray:
    """
    Remove empty borders (all zero) from an oversized Numpy array.

    Args:
        array: Numpy array, dtype=int

    Returns:
        Numpy array of the minimal size that fits non-zero areas.
    """
    ones = np.where(array == 1)
    trimmed = array[min(ones[0]): max(ones[0]) + 1, min(ones[1]): max(ones[1]) + 1]
    return trimmed


def convert_to_image(array: NDArray) -> NDArray:
    """
    Convert ones and zeroes to a more readable form.

    Args:
        array: Numpy array, dtype=int

    Returns:
        Numpy array, dtype=str
    """
    image = array.astype(str)
    image[image == '1'] = '█'
    image[image == '0'] = ' '
    return image


class Point(NamedTuple):
    """A point in a 2d space."""
    row: int
    col: int

    def __add__(self, other: Point) -> Point:
        return Point(self.row + other.row, self.col + other.col)

    def __radd__(self, other: Point) -> Point:
        if other == 0:
            other = Point(0, 0)
        return self.__add__(other)

    def __repr__(self) -> str:
        return f'({self.row}, {self.col})'

    def manhattan_distance(self, other: Point) -> int:
        return abs(self.row - other.row) + abs(self.col - other.col)


def slice_with_complex(arr: NDArray, value: complex) -> Any:
    """
    Return the value at the specified index in the input array.

    Parameters:
        arr: A 2D Numpy array.
        value: An index to slice the array with.

    Returns:
        A value at the specified index in the input array.
    """
    return arr[int(value.real), int(value.imag)]
