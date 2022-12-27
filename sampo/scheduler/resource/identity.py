from typing import List, Callable

import numpy as np

from sampo.scheduler.resource.base import ResourceOptimizer
from sampo.schemas.contractor import WorkerContractorPool
from sampo.schemas.resources import Worker
from sampo.schemas.time import Time


class IdentityResourceOptimizer(ResourceOptimizer):

    def optimize_resources(self,
                           worker_pool: WorkerContractorPool,
                           worker_team: List[Worker],
                           optimize_array: np.ndarray,
                           down_border: np.ndarray,
                           up_border: np.ndarray,
                           get_finish_time: Callable[[List[Worker]], Time]):
        # no actions here
        pass
