from abc import ABC, abstractmethod
from typing import List, Callable, Optional

import numpy as np

from sampo.schemas.contractor import WorkerContractorPool, Contractor
from sampo.schemas.resources import Worker
from sampo.schemas.time import Time


class ResourceOptimizer(ABC):

    @abstractmethod
    def optimize_resources(self,
                           worker_pool: WorkerContractorPool,
                           contractors: List[Contractor],
                           worker_team: List[Worker],
                           optimize_array: Optional[np.ndarray],
                           down_border: np.ndarray,
                           up_border: np.ndarray,
                           get_finish_time: Callable[[List[Worker]], Time]):
        """
        Base resource optimization method. Optimizes `worker_team` using `get_finish_time` metric.
        :param: worker_pool: globally available workers
        :param: contractors: listed contractors
        :param: worker_team: the team to optimize. Optimizing performs in-place
        :param: optimize_array: a boolean array that says what positions should be optimized
        :param: down_border: down-border for resource optimizer
        :param: up_border: up-border for resource optimizer
        :param: get_finish_time: metric
        """
        ...
