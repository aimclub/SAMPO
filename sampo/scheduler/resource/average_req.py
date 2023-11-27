from typing import Callable

import numpy as np

from sampo.scheduler.resource.base import ResourceOptimizer
from sampo.schemas.contractor import WorkerContractorPool
from sampo.schemas.resources import Worker
from sampo.schemas.time import Time


class AverageReqResourceOptimizer(ResourceOptimizer):
    """
    Class that implements optimization the number of resources by counting average resource requirements.
    """

    def __init__(self, k: float = 2):
        """
        :param k: coefficient of average
        """
        self.k = k

    def optimize_resources(self,
                           worker_pool: WorkerContractorPool,
                           worker_team: list[Worker],
                           optimize_array: np.ndarray,
                           down_border: np.ndarray,
                           up_border: np.ndarray,
                           get_finish_time: Callable[[list[Worker]], Time]):
        """
        The resource optimization module, that counts average resource requirements.

        :param worker_pool: global resources pool
        :param worker_team: worker team to optimize
        :param optimize_array: a boolean array that says what positions should be optimized
        :param down_border: down border of optimization
        :param up_border: up border of optimization
        :param get_finish_time: optimization function that should give execution time based on worker team
        """

        if optimize_array:
            for worker, up, down, optimize in zip(worker_team, up_border, down_border, optimize_array):
                if optimize:
                    worker.count = max(1, down) + int((up - down) / self.k)
        else:
            # TODO Remove max()
            for worker, up, down in zip(worker_team, up_border, down_border):
                worker.count = max(1, down) + int((up - down) / self.k)
