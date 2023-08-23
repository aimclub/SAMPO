from typing import Type, Callable

from sampo.scheduler.base import SchedulerType
from sampo.scheduler.generic import GenericScheduler
from sampo.scheduler.heft.prioritization import prioritization
from sampo.scheduler.resource.base import ResourceOptimizer
from sampo.scheduler.resource.full_scan import FullScanResourceOptimizer
from sampo.scheduler.timeline.just_in_time_timeline import JustInTimeTimeline
from sampo.scheduler.timeline.momentum_timeline import MomentumTimeline
from sampo.schemas.time_estimator import WorkTimeEstimator, DefaultWorkEstimator
from sampo.utilities.base_opt import dichotomy_int


class HEFTScheduler(GenericScheduler):
    """
    Scheduler that uses method of a critical path.
    The scheduler gives opportunity to add work only to end.
    """

    def __init__(self,
                 scheduler_type: SchedulerType = SchedulerType.HEFTAddEnd,
                 resource_optimizer: ResourceOptimizer = FullScanResourceOptimizer(),
                 timeline_type: Type = JustInTimeTimeline,
                 work_estimator: WorkTimeEstimator = DefaultWorkEstimator(),
                 prioritization_f: Callable = prioritization,
                 resource_optimize_f: Callable = None):
        if resource_optimize_f is None:
            resource_optimize_f = self.get_default_res_opt_function()
        super().__init__(scheduler_type, resource_optimizer, timeline_type,
                         prioritization_f, resource_optimize_f, work_estimator)
        self._timeline_type = timeline_type


class HEFTBetweenScheduler(HEFTScheduler):
    """
    Type of scheduler that use method of critical path.
    The scheduler give opportunity to add work between existing works.
    """

    def __init__(self,
                 scheduler_type: SchedulerType = SchedulerType.HEFTAddBetween,
                 resource_optimizer: ResourceOptimizer = FullScanResourceOptimizer(),
                 work_estimator: WorkTimeEstimator = DefaultWorkEstimator()):
        super().__init__(scheduler_type, resource_optimizer, MomentumTimeline,
                         resource_optimize_f=self.get_default_res_opt_function(self.get_finish_time),
                         work_estimator=work_estimator)

    @staticmethod
    def get_finish_time(node, worker_team, node2swork, spec, assigned_parent_time, timeline, work_estimator):
        return timeline.find_min_start_time_with_additional(node, worker_team, node2swork, spec, None,
                                                            assigned_parent_time, work_estimator)[1]
