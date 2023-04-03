from abc import ABC, abstractmethod
from typing import Optional

from sampo.schemas.contractor import WorkerContractorPool
from sampo.schemas.time import Time


class WorkTimeEstimator(ABC):
    @abstractmethod
    def set_mode(self, use_idle: Optional[bool] = True, mode: Optional[str] = 'realistic'):
        ...

    @abstractmethod
    def estimate_time(self, work_name: str, work_volume: float, resources: WorkerContractorPool) -> Time:
        ...


# TODO add simple work_time_estimator based on WorkUnit.estimate_static
