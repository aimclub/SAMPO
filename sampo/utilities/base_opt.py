from typing import Callable, Optional

import numpy as np

from sampo.schemas.time import Time

"""
Contains base optimization methods
"""


# float version
def dichotomy_float(down: float, up: float, func, eps: float = 0.000001):
    delta = eps / 5
    while up - down > eps:
        x1 = (down + up - delta) / 2
        x2 = (down + up + delta) / 2
        if func(x1) < func(x2):
            up = x2
        else:
            down = x1

    return down


# int version
def dichotomy_int(down: int, up: int, func: Callable[[int], Time]):
    while up - down > 2:
        x1 = (down + up - 1) >> 1
        x2 = (down + up + 1) >> 1
        if x1 == x2:
            return x1
        if func(x1) < func(x2):
            up = x2
        else:
            down = x1

        # print(str(x1) + ' ' + str(x2) + ' ' + str(down) + ' ' + str(up))

    return (up + down) >> 1


def coordinate_descent(down: np.ndarray, up: np.ndarray,
                       method: Callable[[int, int, Callable[[int], Time]], Time],
                       fitness: Callable[[np.ndarray], Time],
                       optimize_array: Optional[np.ndarray]) -> np.ndarray:
    cur = down.copy()
    for i in range(down.size):
        if optimize_array and not optimize_array[i]:
            continue

        def part(x):
            cur[i] = x
            return fitness(cur)

        cur[i] = method(down[i], up[i], part)
    return cur
