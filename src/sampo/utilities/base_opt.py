from typing import Callable, Optional

import numpy as np

# float version
from sampo.schemas.time import Time


def dichotomy_float(lo: float, hi: float, func, eps: float = 0.000001):
    delta = eps / 5
    while hi - lo > eps:
        x1 = (lo + hi - delta) / 2
        x2 = (lo + hi + delta) / 2
        if func(x1) < func(x2):
            hi = x2
        else:
            lo = x1

    return lo


# int version
def dichotomy_int(lo: int, hi: int, func: Callable[[int], Time]):
    while hi - lo > 2:
        x1 = (lo + hi - 1) >> 1
        x2 = (lo + hi + 1) >> 1
        if x1 == x2:
            return x1
        if func(x1) < func(x2):
            hi = x2
        else:
            lo = x1

        # print(str(x1) + " " + str(x2) + " " + str(lo) + " " + str(hi))

    return (hi + lo) >> 1


def coordinate_descent(lo: np.ndarray, hi: np.ndarray,
                       method: Callable[[int, int, Callable[[int], Time]], Time],
                       fitness: Callable[[np.ndarray], Time],
                       optimize_array: Optional[np.ndarray]) -> np.ndarray:
    cur = lo.copy()
    for i in range(lo.size):
        if optimize_array and not optimize_array[i]:
            continue

        def part(x):
            cur[i] = x
            return fitness(cur)
        cur[i] = method(lo[i], hi[i], part)
    return cur
