from typing import Union, List, Callable, Dict, Any

from schemas.work_estimator import WorkTimeEstimator
from scheduler.base import SchedulerType, Scheduler
from schemas.schedule import Schedule
from scheduler.evolution.base import EvolutionScheduler
from scheduler.topological.base import TopologicalScheduler
from schemas.contractor import Contractor
from schemas.graph import WorkGraph


def get_scheduler_ctor(scheduling_algorithm_type: SchedulerType) -> Callable[[WorkTimeEstimator], Scheduler]:
    if scheduling_algorithm_type is SchedulerType.Topological:
        return TopologicalScheduler
    return EvolutionScheduler


def generate_schedule(scheduling_algorithm_type: SchedulerType,
                      work_time_estimator: WorkTimeEstimator,
                      work_graph: WorkGraph,
                      contractors: Union[Contractor, List[Contractor]],
                      ksg_info: Dict[str, Dict[str, Any]],
                      start: str,
                      validate_schedule: bool) -> Schedule:
    scheduler = get_scheduler_ctor(scheduling_algorithm_type)(work_time_estimator)
    schedule, _ = scheduler.schedule(work_graph,
                                     [contractors] if isinstance(contractors, Contractor) else contractors,
                                     ksg_info,
                                     start,
                                     validate_schedule)
    return schedule
