from typing import List

from sampo.scheduler.base import Scheduler
from sampo.scheduler.resource.average_req import AverageReqResourceOptimizer
from sampo.scheduler.timeline.base import Timeline
from sampo.schemas.contractor import Contractor
from sampo.schemas.exceptions import NoSufficientContractorError
from sampo.schemas.graph import WorkGraph, GraphNode
from sampo.schemas.schedule import Schedule
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.schemas.time import Time


class AverageBinarySearchResourceOptimizingScheduler:

    def __init__(self, base_scheduler: Scheduler):
        self._base_scheduler = base_scheduler
        self._resource_optimizer = AverageReqResourceOptimizer()
        base_scheduler.resource_optimizer = self._resource_optimizer

    def schedule_with_cache(self, wg: WorkGraph, contractors: List[Contractor], deadline: Time,
                            spec: ScheduleSpec = ScheduleSpec(),
                            validate: bool = False, assigned_parent_time: Time = Time(0)) \
            -> tuple[Schedule, Time, Timeline, list[GraphNode]]:
        def call_scheduler(k) -> tuple[Schedule, Time, Timeline, list[GraphNode]]:
            self._resource_optimizer.k = k
            try:
                return self._base_scheduler.schedule_with_cache(wg, contractors, spec, validate, assigned_parent_time)
            except NoSufficientContractorError:
                return None, Time.inf(), None, None

        def fitness(k):
            result = call_scheduler(k)[1]
            if result > deadline:
                result = Time.inf()
            return result

        k_min = 1
        k_max = 10000

        while k_max - k_min > 0.05:
            m = (k_min + k_max) / 2
            time_m = fitness(m)
            if time_m.is_inf():
                k_max = m
            else:
                k_min = m

        return call_scheduler(k_min)
