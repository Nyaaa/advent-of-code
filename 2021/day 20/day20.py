from tools import parsers, loader
import numpy as np
from numpy.typing import NDArray


def parse(data: list[list[str]]) -> tuple[NDArray, NDArray]:
    algorithm = list(data[0][0].replace('.', '0').replace('#', '1'))
    algorithm = np.asarray(algorithm, dtype=np.dtype('u1'))
    image = [list(line.replace('.', '0').replace('#', '1')) for line in data[1]]
    image = np.asarray(image, dtype=np.dtype('u1'))
    return algorithm, image


def enhance(data, steps: int) -> int:
    """test part 1:
    >>> print(enhance(parsers.blocks('test.txt'), 2))
    35

    test part 2:
    >>> print(enhance(parsers.blocks('test.txt'), 50))
    3351
    """
    algorithm, image = parse(data)
    for step in range(1, steps + 1):
        value = 1 if algorithm[0] == 1 and step % 2 == 0 else 0
        image = np.pad(image, pad_width=2, mode='constant', constant_values=value)
        output = np.zeros(shape=image.shape, dtype=np.dtype('u1'))

        for index in np.ndindex(image[1:-1, 1:-1].shape):
            row = index[0] + 1
            col = index[1] + 1
            adjacent = image[row - 1:row + 2, col - 1:col + 2]
            flat = adjacent.ravel().astype(dtype=str)
            f = ''.join(flat)
            output[row][col] = algorithm[int(f, 2)]
        image = output[1:-1, 1:-1]
    return np.count_nonzero(image)


print(enhance(parsers.blocks(loader.get()), 2))  # 5316
print(enhance(parsers.blocks(loader.get()), 50))  # 16728
