from abc import ABC, abstractmethod
from enum import Enum
from operator import attrgetter
from random import Random
from typing import Optional

import numpy.random

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

    def find_work_resources(self, work_name: str, work_volume: float, resource_name: list[str] | None = None) \
            -> list[WorkerReq]:
        if resource_name is None:
            resource_name = ['driver', 'fitter', 'manager', 'handyman', 'electrician', 'engineer']
        return [WorkerReq(kind=name,
                          volume=work_volume * numpy.random.poisson(work_volume ** 0.5, 1)[0],
                          min_count=numpy.random.poisson(work_volume ** 0.2, 1)[0],
                          max_count=numpy.random.poisson(work_volume * 3, 1)[0])
                for name in resource_name]

    def set_estimation_mode(self, use_idle: bool = True, mode: WorkEstimationMode = WorkEstimationMode.Realistic):
        self._use_idle = use_idle
        self._estimation_mode = mode

    def set_productivity_mode(self, mode: WorkerProductivityMode = WorkerProductivityMode.Static):
        self._productivity_mode = mode

    def estimate_static(self, work_unit: WorkUnit, worker_list: list[Worker]) -> Time:
        """
        Calculate summary time of task execution (without stochastic part)

        :param work_unit:
        :param worker_list:
        :return: time of task execution
        """
        # TODO Is it should be here, not in preprocessing???
        # TODO Move it to ksg_scheduling
        # workers = {w.name.replace('_res_fact', ""): w.count for w in worker_list}
        # work_time = self.estimate_time(work_unit.name.split('_stage_')[0], work_unit.volume, workers)
        # if work_time > 0:
        #     return work_time

        return self.estimate_time(work_unit, worker_list)

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
            productivity = DefaultWorkEstimator.get_productivity_of_worker(worker, self.rand, req.max_count,
                                                                           self._productivity_mode) / worker_count
            if productivity == 0:
                return Time.inf()
            times.append(req.volume // productivity)
        return max(max(times), Time(0))

    @staticmethod
    def get_productivity_of_worker(worker: Worker, rand: Optional[Random] = None, max_groups: int = 0,
                                   productivity_mode: WorkerProductivityMode = WorkerProductivityMode.Static):
        """
        Calculate the productivity of the Worker
        It has 2 mods: stochastic and non-stochastic, depending on the value of rand

        :param productivity_mode:
        :param max_groups:
        :param worker: the worker
        :param rand: parameter for stochastic part
        :return: productivity of received worker
        """
        return worker.get_productivity(rand, productivity_mode) * communication_coefficient(worker.count, max_groups)


def communication_coefficient(groups_count: int, max_groups: int) -> float:
    n = groups_count
    m = max_groups
    return 1 / (6 * m ** 2) * (-2 * n ** 3 + 3 * n ** 2 + (6 * m ** 2 - 1) * n)
