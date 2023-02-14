from abc import ABC, abstractmethod
from typing import Optional, Callable

from sampo.schemas.contractor import WorkerContractorPool
from sampo.schemas.time import Time


class WorkTimeEstimator(ABC):
    @abstractmethod
    def set_mode(self, use_idle: Optional[bool] = True, mode: Optional[str] = 'realistic'):
        ...

    @abstractmethod
    def estimate_time(self, work_name: str, work_volume: float, resources: WorkerContractorPool) -> Time:
        ...

    @abstractmethod
    def split_to_constructor_and_params(self) -> tuple[Callable[[...], 'WorkTimeEstimator'], tuple]:
        ...


# TODO add simple work_time_estimator based on WorkUnit.estimate_static
