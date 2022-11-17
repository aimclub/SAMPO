import time
from typing import Union, List, Callable, Dict, Any

from utilities.time_estimator import WorkTimeEstimator
from scheduler.base import SchedulerType, Scheduler
from scheduler.genetic.base import GeneticScheduler
from scheduler.heft.base import HEFTScheduler
from scheduler.heft_between.base import HEFTBetweenScheduler
from scheduler.resource.base import ResourceOptimizer
from scheduler.topological.base import TopologicalScheduler
from schemas.contractor import Contractor
from schemas.graph import WorkGraph
from schemas.schedule import Schedule


def get_scheduler_ctor(scheduling_algorithm_type: SchedulerType) \
        -> Callable[[SchedulerType, ResourceOptimizer, WorkTimeEstimator], Scheduler]:
    if scheduling_algorithm_type is SchedulerType.HEFTAddBetween:
        return HEFTBetweenScheduler
    if scheduling_algorithm_type is SchedulerType.HEFTAddEnd:
        return HEFTScheduler
    if scheduling_algorithm_type is SchedulerType.Topological:
        return TopologicalScheduler
    return GeneticScheduler


def generate_schedule(scheduling_algorithm_type: SchedulerType,
                      work_time_estimator: WorkTimeEstimator,
                      work_graph: WorkGraph,
                      contractors: Union[Contractor, List[Contractor]],
                      start: str,
                      validate_schedule: bool) -> Schedule:
    scheduler = get_scheduler_ctor(scheduling_algorithm_type)(work_estimator=work_time_estimator)
    start_time = time.time()
    schedule = scheduler.schedule(work_graph,
                                  [contractors] if isinstance(contractors, Contractor) else contractors,
                                  start,
                                  validate_schedule)
    print(f'Time: {(time.time() - start_time) * 1000} ms')
    return schedule
