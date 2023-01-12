from abc import abstractmethod, ABC
from collections import defaultdict
from typing import Tuple, List, Optional, Union
from uuid import uuid4

import numpy as np

from sampo.scheduler.base import Scheduler
from sampo.schemas.contractor import Contractor, WorkerContractorPool
from sampo.schemas.graph import WorkGraph
from sampo.schemas.resources import Worker
from sampo.schemas.schedule import ScheduleWorkDict, Schedule
from sampo.schemas.time import Time
from sampo.utilities.visualization.resources import get_workers_intervals, SPLITTER


class ResourceOptimizer(ABC):
    @abstractmethod
    def optimize(self, wg: WorkGraph, deadline: Time) -> Tuple[Contractor, Time]:
        ...


# TODO: remove ScheduleWorkDict option
def max_time_schedule(schedule: Union[ScheduleWorkDict, Schedule]):
    return max(work["finish"] for work in schedule.values()) \
            if isinstance(schedule, ScheduleWorkDict) \
            else max(swork.start_end_time[1] for swork in schedule.works)


def get_schedule_with_time(scheduler: Scheduler, wg: WorkGraph,
                           agent_counts: np.array, agent_names: List[str]) -> Tuple[Schedule, Time]:
    scheduled_works = scheduler.schedule(wg, agents_to_contractors(agent_counts, agent_names))
    max_time = max_time_schedule(scheduled_works)
    return scheduled_works, max_time


def agents_to_contractors(agent_counts: np.array, agent_names: List[str],
                          contractor_id: Optional[str or None] = None,
                          contractor_name: Optional[str] = ""):
    contractor_id = contractor_id or uuid4()
    # TODO Remove...
    # TODO: remove try
    try:
        workers = [Worker(str(uuid4()), name, count) for name, count in zip(agent_names, agent_counts)]
    except Exception:
        print('обосрамс...')
    workers_dict = {(w.name, w.productivity_class): w for w in workers}
    return [Contractor(contractor_id, contractor_name, agent_names, [], workers_dict, {})]


def is_resources_good(wg: WorkGraph,
                      agent_counts: np.array, agent_names: List[str],
                      scheduler: Scheduler, deadline: Time) -> bool:
    try:
        _, schedule_time = get_schedule_with_time(scheduler, wg, agent_counts, agent_names)
        return schedule_time <= deadline
    # TODO: remove
    except AssertionError:
        return False
    except Exception as ex:
        print(ex)
        return False


def find_min_agents(wg: WorkGraph, max_workers: int) -> Tuple[np.array, List[str]]:
    min_agents = defaultdict(lambda: max_workers)
    for node in wg.nodes:
        for req in node.work_unit.worker_reqs:
            min_agents[req.kind] = min(min_agents[req.kind], req.min_count)
    min_counts_list, agent_names = list(zip(*[(count, name) for name, count in min_agents.items()]))
    min_counts: np.array = np.array(min_counts_list, dtype=int)
    return min_counts, agent_names


def get_minimal_counts_by_schedule(scheduled_works: Schedule, agent_names: List[str]) -> np.array:
    workers_intervals = get_workers_intervals(scheduled_works)
    max_used_counts = defaultdict(int)
    for name_index in workers_intervals:
        name, index = name_index.split(SPLITTER)
        max_used_counts[name] = max(max_used_counts[name], int(index))
    optimal_counts = np.array([max_used_counts[name] + 1 for name in agent_names], dtype=int)
    return optimal_counts


def init_borders(wg: WorkGraph, scheduler: Scheduler, deadline: Time,
                 worker_factor: int,
                 max_workers: int,
                 right_agents: WorkerContractorPool or None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
    left_counts, agent_names = find_min_agents(wg, max_workers)
    if right_agents is None:
        right_counts: np.ndarray = left_counts * worker_factor
    else:
        right_counts = np.array([right_agents[name] for name in agent_names])

    if is_resources_good(wg, left_counts, agent_names, scheduler, deadline):
        # wg is resolved in time by minimal set of workers
        return left_counts, None, agent_names

    if not is_resources_good(wg, right_counts, agent_names, scheduler, deadline):
        # wg is not resolved in time by any set of workers
        return None, right_counts, agent_names
    return left_counts, right_counts, agent_names


def prepare_answer(counts: np.array, agent_names: List[str], wg: WorkGraph, scheduler: Scheduler,
                   dry_resources: bool):
    schedule, max_time = get_schedule_with_time(scheduler, wg, counts, agent_names)
    if dry_resources:
        optimal_counts = get_minimal_counts_by_schedule(schedule, agent_names)
    else:
        optimal_counts = counts
    contractor = agents_to_contractors(optimal_counts, agent_names)[0]
    return contractor, max_time
