from abc import ABC, abstractmethod
from typing import List, Callable

import numpy as np

from sampo.schemas.contractor import WorkerContractorPool, Contractor
from sampo.schemas.resources import Worker
from sampo.schemas.time import Time


class ResourceOptimizer(ABC):

    @abstractmethod
    def optimize_resources(self,
                           agents: WorkerContractorPool,
                           contractors: List[Contractor],
                           worker_team: List[Worker],
                           down_border: np.ndarray,
                           up_border: np.ndarray,
                           get_finish_time: Callable[[List[Worker]], Time]):
        ...
