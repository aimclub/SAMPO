from typing import List, Callable

import numpy as np

from sampo.scheduler.resource.base import ResourceOptimizer
from sampo.schemas.contractor import WorkerContractorPool
from sampo.schemas.resources import Worker
from sampo.schemas.time import Time


class AverageReqResourceOptimizer(ResourceOptimizer):

    def __init__(self, k: float = 2):
        self.k = k

    def optimize_resources(self,
                           worker_pool: WorkerContractorPool,
                           worker_team: List[Worker],
                           optimize_array: np.ndarray,
                           down_border: np.ndarray,
                           up_border: np.ndarray,
                           get_finish_time: Callable[[List[Worker]], Time]):

        if optimize_array:
            for i in range(len(worker_team)):
                if optimize_array[i]:
                    worker_team[i].count = max(1, down_border[i]) + int((up_border[i] - down_border[i]) / self.k)
        else:
            # TODO Remove max()
            for i in range(len(worker_team)):
                worker_team[i].count = max(1, down_border[i]) + int((up_border[i] - down_border[i]) / self.k)
