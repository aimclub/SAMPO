from typing import List, Optional, Type

from sampo.scheduler.base import SchedulerType
from sampo.scheduler.generic import GenericScheduler
from sampo.scheduler.heft.prioritization import prioritization
from sampo.scheduler.heft.time_computaion import calculate_working_time_cascade
from sampo.scheduler.resource.base import ResourceOptimizer
from sampo.scheduler.resource.coordinate_descent import CoordinateDescentResourceOptimizer
from sampo.scheduler.timeline.base import Timeline
from sampo.scheduler.timeline.just_in_time_timeline import JustInTimeTimeline
from sampo.scheduler.timeline.momentum_timeline import MomentumTimeline
from sampo.scheduler.utils.multi_contractor import get_worker_borders, run_contractor_search
from sampo.schemas.contractor import Contractor, WorkerContractorPool
from sampo.schemas.graph import GraphNode
from sampo.schemas.resources import Worker
from sampo.schemas.schedule_spec import WorkSpec
from sampo.schemas.scheduled_work import ScheduledWork
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator
from sampo.utilities.base_opt import dichotomy_int


class HEFTScheduler(GenericScheduler):

    def __init__(self,
                 scheduler_type: SchedulerType = SchedulerType.HEFTAddEnd,
                 resource_optimizer: ResourceOptimizer = CoordinateDescentResourceOptimizer(dichotomy_int),
                 timeline_type: Type = JustInTimeTimeline,
                 work_estimator: Optional[WorkTimeEstimator or None] = None):
        super().__init__(scheduler_type, resource_optimizer, timeline_type, prioritization,
                         self.get_default_res_opt_function(), work_estimator)
        self._timeline_type = timeline_type


class HEFTBetweenScheduler(HEFTScheduler):

    def __init__(self,
                 scheduler_type: SchedulerType = SchedulerType.HEFTAddBetween,
                 resource_optimizer: ResourceOptimizer = CoordinateDescentResourceOptimizer(dichotomy_int),
                 work_estimator: Optional[WorkTimeEstimator or None] = None):
        super().__init__(scheduler_type, resource_optimizer, MomentumTimeline, work_estimator)
