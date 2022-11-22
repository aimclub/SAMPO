from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Dict, Any

from external.estimate_time import WorkTimeEstimator
from scheduler.resource.base import ResourceOptimizer
from scheduler.resource.coordinate_descent import CoordinateDescentResourceOptimizer
from schemas.contractor import Contractor
from schemas.schedule import Schedule
from schemas.graph import WorkGraph
from utilities.base_opt import dichotomy_int

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
                 validate_schedule: Optional[bool] = False) \
            -> Schedule:
        ...
