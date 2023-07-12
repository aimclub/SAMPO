from copy import copy
from typing import Iterable

from sampo.scheduler.base import Scheduler
from sampo.scheduler.resource.average_req import AverageReqResourceOptimizer
from sampo.scheduler.timeline.base import Timeline
from sampo.schemas.contractor import Contractor
from sampo.schemas.exceptions import NoSufficientContractorError
from sampo.schemas.graph import WorkGraph, GraphNode
from sampo.schemas.landscape import LandscapeConfiguration
from sampo.schemas.schedule import Schedule
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.schemas.time import Time


def link_works_to_be_independent(wg: WorkGraph, spec: ScheduleSpec) -> Iterable[GraphNode]:
    for node in wg.nodes:
        spec.get_work_spec(node.id).is_independent = True
        yield node

class AverageBinarySearchResourceOptimizingScheduler:
    """
    The scheduler optimizes resources to deadline.
    Scheduler uses binary search to optimize resources.
    """

    def __init__(self, base_scheduler: Scheduler):
        self._base_scheduler = base_scheduler
        self._resource_optimizer = AverageReqResourceOptimizer()
        base_scheduler.resource_optimizer = self._resource_optimizer

    def schedule_with_cache(self, wg: WorkGraph,
                            contractors: list[Contractor],
                            deadline: Time,
                            spec: ScheduleSpec = ScheduleSpec(),
                            validate: bool = False,
                            assigned_parent_time: Time = Time(0),
                            landscape: LandscapeConfiguration = LandscapeConfiguration()) \
            -> tuple[tuple[Schedule, Time, Timeline, list[GraphNode]], ScheduleSpec]:
        def call_scheduler(k: float, inner_spec: ScheduleSpec) \
                -> tuple[tuple[Schedule, Time, Timeline, list[GraphNode]], ScheduleSpec]:
            self._resource_optimizer.k = k
            try:
                return self._base_scheduler.schedule_with_cache(wg, contractors, landscape, inner_spec, validate,
                                                                assigned_parent_time), inner_spec
            except NoSufficientContractorError:
                return None, Time.inf(), None, None, inner_spec

        def fitness(k: float, inner_spec: ScheduleSpec):
            result = call_scheduler(k, inner_spec)[1]
            if result > deadline:
                result = Time.inf()
            return result

        copied_spec = copy(spec)

        k_min = 1
        k_max = 10000

        last_correct = k_max

        try_count = 0
        max_try_count = 3

        result_min_resources = fitness(k_max, copied_spec)
        if result_min_resources < deadline:
            # we can keep the deadline if pass minimum resources,
            # so let's go preventing the works going in parallel
            for node in link_works_to_be_independent(wg, copied_spec):
                new_time = fitness(k_max, copied_spec)
                if new_time > deadline:
                    # if breaking deadline, revert the mutation
                    copied_spec.get_work_spec(node.id).is_independent = False
                    try_count += 1
                    if try_count == max_try_count:
                        # if trying to link independent too many times, break the process
                        break
                else:
                    try_count = 0
        else:
            # we can't keep the deadline if pass minimum resources,
            # but we can pass more

            while k_max - k_min > 0.05:
                m = (k_min + k_max) / 2
                time_m = fitness(m, copied_spec)
                if time_m.is_inf():
                    k_max = m
                else:
                    last_correct = m
                    k_min = m

        return call_scheduler(last_correct, copied_spec)
