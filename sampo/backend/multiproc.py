from enum import Enum, auto
from random import Random
from typing import Callable

import pathos.multiprocessing

from sampo.backend import T, R, ComputationalContext, DefaultComputationalBackend


# logger = pathos.logger()
# logger.setLevel(0)

# context used for caching common data
class MultiprocessingComputationalContext(ComputationalContext):
    def __init__(self, n_cpus: int, initializer: Callable = None, args: tuple = ()):
        self._pool = pathos.multiprocessing.Pool(n_cpus, initializer=initializer, initargs=args)

    def map(self, action: Callable[[T], R], values: list[T]) -> list[R]:
        try:
            return self._pool.map(action, values)
        except Exception as e:
            #logger.debug(e)
            pass


class MultiprocessingComputationalBackend(DefaultComputationalBackend):
    _actions = {}

    def __init__(self, n_cpus: int):
        self._n_cpus = n_cpus
        self._init_chromosomes = None
        super().__init__()

    def new_context(self) -> ComputationalContext:
        return MultiprocessingComputationalContext(self._n_cpus)

    def recreate_pool(self):
        # self.set_context(MultiprocessingComputationalContext(self._n_cpus,
        #                                                      scheduler_info_initializer,
        #                                                      (self._wg,
        #                                                       self._contractors,
        #                                                       self._landscape,
        #                                                       self._spec,
        #                                                       self._population_size,
        #                                                       self._mutate_order,
        #                                                       self._mutate_resources,
        #                                                       self._mutate_zones,
        #                                                       self._init_chromosomes,
        #                                                       self._assigned_parent_time,
        #                                                       self._rand,
        #                                                       self._work_estimator.get_recreate_info())))
        pass
