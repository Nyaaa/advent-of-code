import numpy as np
from numba import njit
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray

from tools import loader, parsers


@njit
def rolling_window(array: NDArray, window_shape: tuple[int, int]) -> NDArray:
    windows = sliding_window_view(array, window_shape)
    return np.asarray([[np.sum(i) for i in j] for j in windows])


class Fuel:
    def __init__(self, data: str) -> None:
        self.serial = int(data)
        self.grid: NDArray = np.fromfunction(self.get_power_level, (300, 300), dtype=int)

    def get_power_level(self, x: int, y: int) -> int:
        """
        >>> print(Fuel('57').get_power_level(122, 79))
        -5"""
        cell_id = x + 10
        power = (cell_id * y + self.serial) * cell_id
        return (power // 100 % 10) - 5

    def part_1(self) -> tuple[int, ...]:
        """
        >>> print(Fuel('18').part_1())
        (33, 45)"""
        sums = rolling_window(self.grid, (3, 3))
        return np.unravel_index(sums.argmax(), sums.shape)

    def part_2(self) -> tuple[int, ...]:
        """
        >>> print(Fuel('18').part_2())
        (90, 269, 16)"""
        best = 0
        result = ()
        no_increase = 0
        for i in range(3, 301):
            sums = rolling_window(self.grid, (i, i))
            value = np.max(sums)
            if value > best:
                best = value
                result = np.unravel_index(sums.argmax(), sums.shape) + (i, )
                no_increase = 0
            elif value <= best:
                no_increase += 1
            if no_increase > 3:  # In theory, some inputs may not work with this optimization
                break
        return result


print(Fuel(parsers.string(loader.get())).part_1())  # 243, 27
print(Fuel(parsers.string(loader.get())).part_2())  # 284, 172, 12
