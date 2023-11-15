from copy import deepcopy

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
                return (None, Time.inf(), None, None), inner_spec

        def fitness(k: float, inner_spec: ScheduleSpec):
            result = call_scheduler(k, inner_spec)[0][0].execution_time
            # if result > deadline:
            #     result = Time.inf()
            return result

        copied_spec = deepcopy(spec)
        # FIXME Investigate why `spec` given to this method can be saved from previous call and remove this
        copied_spec._work2spec.clear()

        k_min = 1
        k_max = 10000

        best = k_max

        try_count = 0
        max_try_count = 3

        result_min_resources = fitness(k_max, copied_spec)
        if result_min_resources < deadline:
            # print('Can keep deadline at minimum resources')
            # we can keep the deadline if pass minimum resources,
            # so let's go preventing the works going in parallel
            for node in wg.nodes:
                copied_spec.get_work_spec(node.id).is_independent = True
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
            return call_scheduler(best, copied_spec)
        else:
            # we can't keep the deadline if pass minimum resources,
            # but we can pass more

            best_left_fit = Time.inf()
            best_right_fit = Time.inf()
            best_left_m = k_max
            best_right_m = k_min

            while k_max - k_min > 0.05:
                m = (k_min + k_max) / 2
                time_m = fitness(m, copied_spec)
                if time_m > deadline:
                    if abs(time_m.value - deadline.value) <= best_right_fit:
                        best_right_fit = abs(time_m.value - deadline.value)
                        best_right_m = m
                    k_max = m
                else:
                    if abs(time_m.value - deadline.value) <= best_left_fit:
                        best_left_fit = abs(time_m.value - deadline.value)
                        best_left_m = m
                    k_min = m

            if best_left_fit < Time.inf():
                return call_scheduler(best_left_m, copied_spec)
            return call_scheduler(best_right_m, copied_spec)
