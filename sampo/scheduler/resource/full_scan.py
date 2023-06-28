from typing import Callable

import numpy as np

from sampo.scheduler.resource.base import ResourceOptimizer
from sampo.scheduler.resource.coordinate_descent import CoordinateDescentResourceOptimizer
from sampo.schemas.contractor import WorkerContractorPool
from sampo.schemas.resources import Worker
from sampo.schemas.time import Time
from sampo.utilities.base_opt import dichotomy_int


class FullScanResourceOptimizer(ResourceOptimizer):
    """
    Class that implements optimization the number of resources by the smart search method.
    """

    _coordinate_descent_optimizer = CoordinateDescentResourceOptimizer(dichotomy_int)

    def optimize_resources(self,
                           worker_pool: WorkerContractorPool,
                           worker_team: list[Worker],
                           optimize_array: np.ndarray,
                           down_border: np.ndarray,
                           up_border: np.ndarray,
                           get_finish_time: Callable[[list[Worker]], Time]):
        """
        The resource optimization module, that search optimal number of resources by the smart search method.

        :param worker_pool: global resources pool
        :param worker_team: worker team to optimize
        :param optimize_array: a boolean array that says what positions should be optimized
        :param down_border: down border of optimization
        :param up_border: up border of optimization
        :param get_finish_time: optimization function that should give execution time based on worker team
        """

        # TODO Handle optimize_array
        def fitness(worker_count: np.ndarray):
            """
            Function that counts finish time with a current set of resources

            :param worker_count:
            :return: finish time with a current set of resources
            """
            for ind, worker in enumerate(worker_team):
                worker.count = worker_count[ind]
            return get_finish_time(worker_team)

        right_fit = fitness(up_border)
        left_fit = fitness(down_border)

        mid = up_border

        # Find the optimal point on the search area diagonal
        while (mid > down_border).any():
            mid = (up_border + down_border) // 2
            next_fit = fitness(mid)
            if next_fit < right_fit:
                down_border = mid
                left_fit = next_fit
            elif next_fit > left_fit:
                up_border = mid
                right_fit = next_fit
            else:
                break

        # Use coordinate decent to search an optimal set of resources with new down_border
        self._coordinate_descent_optimizer.optimize_resources(worker_pool,
                                                              worker_team,
                                                              optimize_array,
                                                              mid,
                                                              up_border,
                                                              get_finish_time)
