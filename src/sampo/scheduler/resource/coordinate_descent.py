from typing import List, Callable

import numpy as np

from sampo.scheduler.resource.base import ResourceOptimizer
from sampo.schemas.contractor import WorkerContractorPool
from sampo.schemas.resources import Worker
from sampo.schemas.time import Time
from sampo.utilities.base_opt import coordinate_descent


class CoordinateDescentResourceOptimizer(ResourceOptimizer):

    def __init__(self, one_dimension_optimizer: Callable[[int, int, Callable[[int], Time]], Time]):
        self.one_dimension_optimizer = one_dimension_optimizer

    def optimize_resources(self,
                           worker_pool: WorkerContractorPool,
                           worker_team: List[Worker],
                           optimize_array: np.ndarray,
                           down_border: np.ndarray,
                           up_border: np.ndarray,
                           get_finish_time: Callable[[List[Worker]], Time]):

        def fitness(worker_count: np.ndarray):
            for worker_ind in range(len(worker_team)):
                worker_team[worker_ind].count = worker_count[worker_ind]
            return get_finish_time(worker_team)

        count_worker_team = coordinate_descent(down_border, up_border,
                                               self.one_dimension_optimizer,
                                               fitness,
                                               optimize_array)
        if optimize_array:
            for i in range(len(worker_team)):
                if optimize_array[i]:
                    worker_team[i].count = count_worker_team[i]
        else:
            for i in range(len(worker_team)):
                worker_team[i].count = count_worker_team[i]
