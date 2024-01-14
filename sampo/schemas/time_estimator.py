from abc import ABC, abstractmethod
from enum import Enum
from operator import attrgetter
from random import Random
from typing import Optional

import numpy.random
import math

from sampo.schemas import Interval, IntervalGaussian
from sampo.schemas.requirements import WorkerReq
from sampo.schemas.resources import Worker
from sampo.schemas.resources import WorkerProductivityMode
from sampo.schemas.time import Time
from sampo.schemas.works import WorkUnit
from sampo.utilities.collections_util import build_index


class WorkEstimationMode(Enum):
    Pessimistic = -1,
    Realistic = 0,
    Optimistic = 1


class WorkTimeEstimator(ABC):
    """
    Implementation of time estimator of work with a given set of resources.
    """

    @abstractmethod
    def set_estimation_mode(self, use_idle: bool = True, mode: WorkEstimationMode = WorkEstimationMode.Realistic):
        ...

    @abstractmethod
    def set_productivity_mode(self, mode: WorkerProductivityMode = WorkerProductivityMode.Static):
        ...

    @abstractmethod
    def find_work_resources(self, work_name: str, work_volume: float, resource_name: list[str] | None = None) \
            -> list[WorkerReq]:
        ...

    @abstractmethod
    def estimate_time(self, work_unit: WorkUnit, worker_list: list[Worker]):
        ...


class DefaultWorkEstimator(WorkTimeEstimator):

    def __init__(self,
                 rand: Random = Random()):
        self._use_idle = True
        self._estimation_mode = WorkEstimationMode.Realistic
        self.rand = rand
        self._productivity_mode = WorkerProductivityMode.Static
        self._productivity = {worker: {'__ALL__': IntervalGaussian(1, 0.2, 1, 0)}
                              for worker in ['driver', 'fitter', 'manager', 'handyman', 'electrician', 'engineer']}

    def find_work_resources(self, work_name: str, work_volume: float, resource_name: list[str] | None = None) \
            -> list[WorkerReq]:
        if resource_name is None:
            resource_name = ['driver', 'fitter', 'manager', 'handyman', 'electrician', 'engineer']
        dist = numpy.random.poisson(work_volume * 3, len(resource_name))
        return [WorkerReq(kind=name,
                          volume=Time(math.ceil(work_volume * numpy.random.poisson(work_volume ** 0.5, 1)[0])),
                          min_count=int(dist[i]),
                          max_count=int(dist[i] * 2))
                for i, name in enumerate(resource_name)]

    def set_estimation_mode(self, use_idle: bool = True, mode: WorkEstimationMode = WorkEstimationMode.Realistic):
        self._use_idle = use_idle
        self._estimation_mode = mode

    def set_productivity_mode(self, mode: WorkerProductivityMode = WorkerProductivityMode.Static):
        self._productivity_mode = mode

    def estimate_time(self, work_unit: WorkUnit, worker_list: list[Worker]) -> Time:
        if not worker_list:
            return Time(0)

        times = [Time(0)]  # if there are no requirements for the work, it is done instantly
        name2worker = build_index(worker_list, attrgetter('name'))

        for req in work_unit.worker_reqs:
            if req.min_count == 0:
                continue
            name = req.kind
            worker = name2worker.get(name, None)
            worker_count = 0 if worker is None else worker.count
            if worker_count < req.min_count:
                return Time.inf()
            productivity = self._get_productivity(worker, req.max_count)
            if productivity == 0:
                return Time.inf()
            times.append(Time(math.ceil(req.volume / productivity)))
        return max(max(times), Time(0))

    def _get_productivity(self, worker: Worker, max_count_workers: int) -> float:
        """
        Calculate the productivity of the Worker
        It has 2 modes: stochastic and non-stochastic, depending on the value of rand
        :param worker: the worker
        :param max_count_workers: maximum workers count according to worker reqs
        :return:
        """
        worker_productivities = self._productivity[worker.name]
        productivity_interval = worker_productivities.get(worker.contractor_id, worker_productivities['__ALL__'])
        productivity = productivity_interval.mean \
            if self._productivity_mode is WorkerProductivityMode.Static \
            else productivity_interval.rand_float(self.rand)
        return productivity * worker.count * communication_coefficient(worker.count, max_count_workers)

    def set_worker_productivity(self, productivity: Interval, name: str, contractor: str | None = None):
        if contractor is None:
            self._productivity[name]['__ALL__'] = productivity
            return
        self._productivity[name][contractor] = productivity


def communication_coefficient(groups_count: int, max_groups: int) -> float:
    n = groups_count
    m = max_groups
    return 1.0 / (6 * m ** 2) * (-2 * n ** 3 + 3 * n ** 2 + (6 * m ** 2 - 1) * n)
