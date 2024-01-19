from abc import ABC

import pathos.multiprocessing
from typing import Callable

from sampo.backend import ComputationalBackend, T, R, ComputationalContext

# context used for caching common data
class MultiprocessingComputationalContext(ComputationalContext):
    def __init__(self, n_cpus: int, initializer: Callable = None):
        self._pool = pathos.multiprocessing.Pool(n_cpus, initializer=initializer)

    def map(self, action: Callable[[T], R], values: list[T]) -> list[R]:
        return self._pool.map(action, values)


class MultiprocessingComputationalBackend(ComputationalBackend, ABC):
    def __init__(self, n_cpus: int):
        super().__init__()
        self.set_context(MultiprocessingComputationalContext(n_cpus))
