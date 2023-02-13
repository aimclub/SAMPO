from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional, Callable

import numpy as np

from sampo.scheduler.resource.base import ResourceOptimizer
from sampo.scheduler.resource.coordinate_descent import CoordinateDescentResourceOptimizer
from sampo.scheduler.timeline.base import Timeline
from sampo.schemas.contractor import Contractor
from sampo.schemas.graph import WorkGraph
from sampo.schemas.resources import Worker
from sampo.schemas.schedule import Schedule
from sampo.schemas.schedule_spec import ScheduleSpec, WorkSpec
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator
from sampo.schemas.works import WorkUnit
from sampo.utilities.base_opt import dichotomy_int


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

    def __str__(self):
        return str(self.scheduler_type.name)

    def schedule(self, wg: WorkGraph,
                 contractors: List[Contractor],
                 spec: ScheduleSpec = ScheduleSpec(),
                 validate: bool = False,
                 start_time: Time = Time(0),
                 timeline: Timeline | None = None) \
            -> Schedule:
        if wg is None or len(wg.nodes) == 0:
            raise ValueError('None or empty WorkGraph')
        if contractors is None or len(contractors) == 0:
            raise ValueError('None or empty contractor list')
        schedule = self.schedule_with_cache(wg, contractors, spec, validate, start_time, timeline)[0]
        # print(f'Schedule exec time: {schedule.execution_time} days')
        return schedule

    @abstractmethod
    def schedule_with_cache(self, wg: WorkGraph,
                            contractors: List[Contractor],
                            spec: ScheduleSpec = ScheduleSpec(),
                            validate: bool = False,
                            assigned_parent_time: Time = Time(0),
                            timeline: Timeline | None = None) \
            -> tuple[Schedule, Time, Timeline]:
        ...

    @staticmethod
    def optimize_resources_using_spec(work_unit: WorkUnit, worker_team: List[Worker], work_spec: WorkSpec,
                                      optimize_lambda: Callable[[np.ndarray], None] = lambda _: None):
        """
        Applies worker team spec to optimization process.
        Can use arbitrary heuristics to increase spec handling efficiency.

        :param work_unit: current work unit
        :param worker_team: current worker team from chosen contractor
        :param work_spec: spec for given work unit
        :param optimize_lambda: optimization func that should hold optimization
            data in its closure and run optimization process when receives `optimize_array`.
            Passing None or default value means this function should only apply spec.
        """
        if len(work_spec.assigned_workers) == len(work_unit.worker_reqs):
            # all resources passed in spec, skipping optimize_resources step
            for w in worker_team:
                w.count = work_spec.assigned_workers[w.name]
        else:
            # create optimize array to save optimizing time
            # this array should contain True if position should be optimized or False if shouldn't
            optimize_array = None
            if work_spec.assigned_workers:
                optimize_array = []
                for w in worker_team:
                    spec_count = work_spec.assigned_workers.get(w.name, 0)
                    if spec_count > 0:
                        w.count = spec_count
                        optimize_array.append(False)
                    else:
                        optimize_array.append(True)

                optimize_array = np.array(optimize_array)

            optimize_lambda(optimize_array)
