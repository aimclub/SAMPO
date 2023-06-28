from typing import Callable

import numpy as np

from sampo.scheduler.resource.base import ResourceOptimizer
from sampo.schemas.contractor import WorkerContractorPool
from sampo.schemas.resources import Worker
from sampo.schemas.time import Time
from sampo.utilities.base_opt import coordinate_descent


class CoordinateDescentResourceOptimizer(ResourceOptimizer):
    """
    Class that implements optimization the number of resources by discrete analogue of coordinate descent.
    """

    def __init__(self, one_dimension_optimizer: Callable[[int, int, Callable[[int], Time]], Time]):
        self.one_dimension_optimizer = one_dimension_optimizer

    def optimize_resources(self,
                           worker_pool: WorkerContractorPool,
                           worker_team: list[Worker],
                           optimize_array: np.ndarray,
                           down_border: np.ndarray,
                           up_border: np.ndarray,
                           get_finish_time: Callable[[list[Worker]], Time]):
        """
        The resource optimization module, that search optimal number of resources by coordinate descent.

        :param worker_pool: global resources pool
        :param worker_team: worker team to optimize
        :param optimize_array: a boolean array that says what positions should be optimized
        :param down_border: down border of optimization
        :param up_border: up border of optimization
        :param get_finish_time: optimization function that should give execution time based on the worker team
        """

        def fitness(worker_count: np.ndarray):
            """
            Function that counts finish time with a current set of resources.

            :param worker_count:
            :return:
            """
            for ind, worker in enumerate(worker_team):
                worker.count = worker_count[ind]
            return get_finish_time(worker_team)

        count_worker_team = coordinate_descent(down_border, up_border,
                                               self.one_dimension_optimizer,
                                               fitness,
                                               optimize_array)
        if optimize_array:
            for i, worker in enumerate(worker_team):
                if optimize_array[i]:
                    worker.count = count_worker_team[i]
        else:
            for i, worker in enumerate(worker_team):
                worker.count = count_worker_team[i]
