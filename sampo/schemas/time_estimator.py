from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from random import Random
from typing import Callable

from sampo.schemas.resources import Worker
from sampo.schemas.time import Time
from sampo.schemas.works import WorkUnit


class WorkEstimationMode(Enum):
    Pessimistic = -1,
    Realistic = 0,
    Optimistic = 1


class WorkTimeEstimator(ABC):
    @abstractmethod
    def set_mode(self, use_idle: bool = True, mode: WorkEstimationMode = WorkEstimationMode.Realistic):
        ...

    @abstractmethod
    def find_work_resources(self, work_name: str, work_volume: float) -> dict[str, int]:
        ...
    
    @abstractmethod
    def estimate_time(self, work_unit: WorkUnit, resources: list[Worker], rand: Random | None = None):
        ...


# TODO add simple work_time_estimator based on WorkUnit.estimate_static
class AbstractWorkEstimator(WorkTimeEstimator, ABC):

    def __init__(self,
                 get_worker_productivity: Callable[[Worker, Random], float],
                 get_team_productivity_modifier: Callable[[int, int], float]):
        self._use_idle = True
        self._mode = WorkEstimationMode.Realistic
        self._get_worker_productivity = get_worker_productivity
        self._get_team_productivity_modifier = get_team_productivity_modifier

    def set_mode(self, use_idle: bool = True, mode: WorkEstimationMode = WorkEstimationMode.Realistic):
        self._use_idle = use_idle
        self._mode = mode

    def estimate_time(self, work_unit: WorkUnit, resources: list[Worker], rand: Random | None = None) -> Time:
        groups = defaultdict(Worker)
        for w in resources:
            groups[w.name] = w
        times = [Time(0)]  # if there are no requirements for the work, it is done instantly
        for req in work_unit.worker_reqs:
            if req.min_count == 0:
                continue
            name = req.kind
            worker = groups[name]
            if worker.count < req.min_count:
                return Time.inf()
            productivity = self._get_worker_productivity(worker, rand) / worker.count \
                         * self._get_team_productivity_modifier(worker.count, req.max_count)
            if productivity == 0:
                return Time.inf()
            times.append(req.volume // productivity)
        return max(times)
