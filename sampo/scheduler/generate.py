import time
from typing import Union, List, Callable

from sampo.scheduler.base import SchedulerType, Scheduler
from sampo.scheduler.genetic.base import GeneticScheduler
from sampo.scheduler.heft.base import HEFTBetweenScheduler
from sampo.scheduler.heft.base import HEFTScheduler
from sampo.scheduler.resource.base import ResourceOptimizer
from sampo.scheduler.topological.base import TopologicalScheduler
from sampo.schemas.contractor import Contractor
from sampo.schemas.graph import WorkGraph
from sampo.schemas.schedule import Schedule
from sampo.schemas.time_estimator import WorkTimeEstimator
from sampo.utilities.resource_cost import schedule_cost


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
                      validate_schedule: bool) -> Schedule:
    scheduler = get_scheduler_ctor(scheduling_algorithm_type)(work_estimator=work_time_estimator)
    start_time = time.time()
    if isinstance(scheduler, GeneticScheduler):
        scheduler.set_use_multiprocessing(n_cpu=10)
    schedule = scheduler.schedule(work_graph,
                                  [contractors] if isinstance(contractors, Contractor) else contractors,
                                  validate=validate_schedule)
    print(f'Time: {(time.time() - start_time) * 1000} ms')
    print(f'Cost: {schedule_cost(schedule)}')
    return schedule
