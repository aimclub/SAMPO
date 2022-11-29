from abc import ABC, abstractmethod
from typing import List, Callable, Optional

import numpy as np

from sampo.schemas.contractor import WorkerContractorPool
from sampo.schemas.resources import Worker
from sampo.schemas.time import Time


class ResourceOptimizer(ABC):

    @abstractmethod
    def optimize_resources(self,
                           worker_pool: WorkerContractorPool,
                           worker_team: List[Worker],
                           optimize_array: Optional[np.ndarray],
                           down_border: np.ndarray,
                           up_border: np.ndarray,
                           get_finish_time: Callable[[List[Worker]], Time]):
        """
        The resource optimization module. Optimizes `worker_team` using `get_finish_time` metric. Should optimize `worker_team` in-place.
        :param worker_pool: global resources pool
        :param worker_team: worker team to optimize
        :param optimize_array: a boolean array that says what positions should be optimized
        :param down_border: down border of optimization
        :param up_border: up border of optimization
        :param get_finish_time: optimization function that should give execution time based on worker team
        """
        ...
