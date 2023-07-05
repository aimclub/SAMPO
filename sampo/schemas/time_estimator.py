from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum
from random import Random
from typing import Optional

from sampo.schemas.resources import Worker
from sampo.schemas.time import Time
from sampo.schemas.works import WorkUnit


class WorkEstimationMode(Enum):
    Pessimistic = -1,
    Realistic = 0,
    Optimistic = 1


class WorkTimeEstimator(ABC):
    """
    Implementation of time estimator of work with a given set of resources.
    """

    @abstractmethod
    def set_mode(self, use_idle: bool = True, mode: WorkEstimationMode = WorkEstimationMode.Realistic):
        ...

    @abstractmethod
    def find_work_resources(self, work_name: str, work_volume: float) -> dict[str, int]:
        ...

    @abstractmethod
    def estimate_time(self, work_unit: WorkUnit, worker_list: list[Worker]):
        ...


# TODO add simple work_time_estimator based on WorkUnit.estimate_static
class AbstractWorkEstimator(WorkTimeEstimator, ABC):

    def __init__(self,
                 rand: Random | None = None):
        self._use_idle = True
        self._mode = WorkEstimationMode.Realistic
        self.rand = rand

    def set_mode(self, use_idle: bool = True, mode: WorkEstimationMode = WorkEstimationMode.Realistic):
        self._use_idle = use_idle
        self._mode = mode

    def estimate_time(self, work_unit: WorkUnit, worker_list: list[Worker]) -> Time:
        groups = defaultdict(Worker)
        for w in worker_list:
            groups[w.name] = w
        times = [Time(0)]  # if there are no requirements for the work, it is done instantly
        for req in work_unit.worker_reqs:
            if req.min_count == 0:
                continue
            name = req.kind
            worker = groups[name]
            if worker.count < req.min_count:
                return Time.inf()
            productivity = self.get_productivity_of_worker(worker, self.rand, req.max_count) / worker.count
            if productivity == 0:
                return Time.inf()
            times.append(req.volume // productivity)
        return max(max(times), Time(1))

    @staticmethod
    def get_productivity_of_worker(worker: Worker, rand: Optional[Random] = None, max_groups: int = 0):
        """
        Calculate the productivity of the Worker
        It has 2 mods: stochastic and non-stochastic, depending on the value of rand

        :param max_groups:
        :param worker: the worker
        :param rand: parameter for stochastic part
        :return: productivity of received worker
        """
        return worker.get_productivity(rand) * communication_coefficient(worker.count, max_groups)


def communication_coefficient(groups_count: int, max_groups: int) -> float:
    n = groups_count
    m = max_groups
    return 1 / (6 * m ** 2) * (-2 * n ** 3 + 3 * n ** 2 + (6 * m ** 2 - 1) * n)
