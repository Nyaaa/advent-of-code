import re

import numpy as np
from numpy.typing import NDArray

from tools import common, loader, parsers

np.set_printoptions(threshold=np.inf, linewidth=np.inf)


def stars(data: list[str]) -> tuple[NDArray, int]:
    data = np.asarray([[int(i) for i in re.findall(r'-?\d+', line)] for line in data])
    locations = data[:, :2]
    vectors = data[:, 2:]
    result = locations.copy()
    min_size = float('inf')
    time = 0

    for time in range(15000):  # noqa: B007
        size = np.amax(locations[:, 1]) - np.amin(locations[:, 1])
        if size <= min_size:
            min_size = size
            result = locations.copy()
        else:
            break
        locations += vectors

    message = np.zeros(result.max(axis=0) + 2, dtype=int)
    message[result[:, 0], result[:, 1]] += 1
    message = np.rot90(message)
    message = np.flipud(message)
    message = common.trim_array(message)
    message = common.convert_to_image(message)
    return message, time - 1


print(stars(parsers.lines(loader.get())))  # EKALLKLB, 10227
