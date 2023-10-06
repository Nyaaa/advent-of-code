import textwrap

import numpy as np
from numpy.typing import NDArray

from tools import common, loader, parsers

np.set_printoptions(linewidth=np.inf)


class SIF:
    def __init__(self, data: list, width: int, height: int) -> None:
        self.layers = [np.asarray(list(layer), dtype=int).reshape((height, width))
                       for layer in textwrap.wrap(data[0], width * height)]

    def part_1(self) -> int:
        zeroes = [np.count_nonzero(layer == 0) for layer in self.layers]
        m = min(i for i in zeroes if i > 0)
        layer = self.layers[zeroes.index(m)]
        return np.count_nonzero(layer == 1) * np.count_nonzero(layer == 2)

    def get_non_transparent(self, index: int) -> int:
        for i in self.layers[1:]:
            pixel = i[index]
            if pixel != 2:
                return pixel

    def part_2(self) -> NDArray:
        image = self.layers[0]
        while 2 in image:
            for i, val in np.ndenumerate(image):
                if val == 2:
                    x = self.get_non_transparent(i)
                    image[i] = x
        return common.convert_to_image(image)


print(SIF(parsers.lines(loader.get()), 25, 6).part_1())  # 2032
print(SIF(parsers.lines(loader.get()), 25, 6).part_2())  # CFCUG
