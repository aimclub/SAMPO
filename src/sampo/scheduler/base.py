from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional

from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.schemas.time_estimator import WorkTimeEstimator
from sampo.scheduler.resource.base import ResourceOptimizer
from sampo.scheduler.resource.coordinate_descent import CoordinateDescentResourceOptimizer
from sampo.schemas.contractor import Contractor
from sampo.schemas.schedule import Schedule
from sampo.schemas.graph import WorkGraph
from sampo.utilities.base_opt import dichotomy_int

TIME_SHIFT = 0.05


class SchedulerType(Enum):
    Topological = 'topological'
    HEFTAddEnd = 'heft_add_end'
    HEFTAddBetween = 'heft_add_between'
    Genetic = 'genetic'


class Scheduler(ABC):
    scheduler_type: SchedulerType
    resource_optimizer: ResourceOptimizer

    def __init__(self,
                 scheduler_type: SchedulerType,
                 resource_optimizer: ResourceOptimizer = CoordinateDescentResourceOptimizer(dichotomy_int),
                 work_estimator: Optional[WorkTimeEstimator] = None):
        self.scheduler_type = scheduler_type
        self.resource_optimizer = resource_optimizer
        self.work_estimator = work_estimator

    @abstractmethod
    def schedule(self, wg: WorkGraph,
                 contractors: List[Contractor],
                 spec: ScheduleSpec = ScheduleSpec(),
                 validate_schedule: Optional[bool] = False) \
            -> Schedule:
        ...
