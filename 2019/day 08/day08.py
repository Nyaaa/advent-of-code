from tools import parsers, loader, common
import textwrap
import numpy as np
from numpy.typing import NDArray

np.set_printoptions(linewidth=np.inf)


class SIF:
    def __init__(self, data: list, width: int, height: int):
        self.layers: list[NDArray] = []
        for layer in textwrap.wrap(data[0], width * height):
            layer = np.asarray(list(layer), dtype=int)
            layer = layer.reshape((height, width))
            self.layers.append(layer)

    def part_1(self):
        zeroes = [np.count_nonzero(layer == 0) for layer in self.layers]
        m = min(i for i in zeroes if i > 0)
        layer = self.layers[zeroes.index(m)]
        return np.count_nonzero(layer == 1) * np.count_nonzero(layer == 2)

    def get_non_transparent(self, index):
        for i in self.layers[1:]:
            pixel = i[index]
            if pixel != 2:
                return pixel

    def part_2(self):
        image = self.layers[0]
        while 2 in image:
            for i, val in np.ndenumerate(image):
                if val == 2:
                    image[i] = self.get_non_transparent(i)
        return common.convert_to_image(image)


print(SIF(parsers.lines(loader.get()), 25, 6).part_1())  # 2032
print(SIF(parsers.lines(loader.get()), 25, 6).part_2())  # CFCUG
