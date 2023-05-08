"""A collection of common functions for AoC."""
from typing import Iterator, Any
from numpy.typing import NDArray
from numba import njit


@njit()
def get_adjacent(
        array: NDArray,
        position: tuple[int, int],
        with_corners: bool = False,
        with_self: bool = False) -> Iterator[tuple[tuple[int, int], Any]]:
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
