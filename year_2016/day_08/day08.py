import numpy as np
from numpy.typing import NDArray

from tools import common, loader, parsers

np.set_printoptions(linewidth=500)
TEST = """rect 3x2
rotate column x=1 by 1
rotate row y=0 by 4
rotate column x=1 by 1"""


def screen(data: list[str]) -> tuple[NDArray, int]:
    arr = np.zeros((6, 50), dtype=np.dtype('u1'))
    for line in data:
        line = line.split()
        match line:
            case 'rect', val:
                a, b = val.split('x')
                arr[0:int(b), 0:int(a)] = 1
            case 'rotate', what, a, _, b:
                a = int(a.split('=')[-1])
                _slice = np.s_[:, a] if what == 'column' else np.s_[a, :]
                arr[_slice] = np.roll(arr[_slice], int(b))
    image = common.convert_to_image(arr)
    return image, np.count_nonzero(arr)


print(screen(parsers.lines(loader.get())))  # 121, RURUCEOEIL
