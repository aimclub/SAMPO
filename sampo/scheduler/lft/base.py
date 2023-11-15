from typing import Type

from sampo.scheduler.heft.base import HEFTScheduler
from sampo.scheduler.base import SchedulerType
from sampo.scheduler.resource.base import ResourceOptimizer
from sampo.scheduler.resource.full_scan import FullScanResourceOptimizer
from sampo.scheduler.timeline.just_in_time_timeline import JustInTimeTimeline
from sampo.schemas.time_estimator import WorkTimeEstimator, DefaultWorkEstimator
from sampo.scheduler.lft.prioritization import lft_prioritization


class LFTScheduler(HEFTScheduler):
    """

    """

    def __init__(self,
                 scheduler_type: SchedulerType = SchedulerType.HEFTAddEnd,
                 resource_optimizer: ResourceOptimizer = FullScanResourceOptimizer(),
                 timeline_type: Type = JustInTimeTimeline,
                 work_estimator: WorkTimeEstimator = DefaultWorkEstimator()):
        super().__init__(scheduler_type, resource_optimizer, timeline_type, work_estimator, lft_prioritization)

