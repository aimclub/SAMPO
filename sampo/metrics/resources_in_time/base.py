from abc import abstractmethod, ABC
from collections import defaultdict
from typing import Optional, Union
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
    def optimize(self, wg: WorkGraph, deadline: Time) -> tuple[Contractor, Time]:
        ...


# TODO: remove ScheduleWorkDict option
def max_time_schedule(schedule: Union[ScheduleWorkDict, Schedule]):
    return max(work["finish"] for work in schedule.values()) \
        if isinstance(schedule, ScheduleWorkDict) \
        else max(swork.start_end_time[1] for swork in schedule.works)


def get_schedule_with_time(scheduler: Scheduler, wg: WorkGraph,
                           worker_counts: np.array, worker_names: list[str]) -> tuple[Schedule, Time]:
    """
    Build schedule and return it with max execution time

    :param scheduler:
    :param wg: received WorkGraph
    :param worker_counts:
    :param worker_names:
    :return:
    """
    scheduled_works = scheduler.schedule(wg, workers_to_contractors(worker_counts, worker_names))
    max_time = max_time_schedule(scheduled_works)
    return scheduled_works, max_time


def workers_to_contractors(worker_counts: np.array,
                           worker_names: list[str],
                           contractor_id: Optional[str or None] = None,
                           contractor_name: Optional[str] = ""):
    contractor_id = contractor_id or uuid4()
    workers = [Worker(str(uuid4()), name, count) for name, count in zip(worker_names, worker_counts)]
    workers_dict = {(w.name, w.productivity_class): w for w in workers}
    return [Contractor(contractor_id, contractor_name, worker_names, [], workers_dict, {})]


def is_resources_good(wg: WorkGraph,
                      worker_counts: np.array,
                      worker_names: list[str],
                      scheduler: Scheduler,
                      deadline: Time) -> bool:
    """
    Return if schedule with given set of resources will be done before deadline

    :param wg:
    :param worker_counts:
    :param worker_names:
    :param scheduler:
    :param deadline:
    :return:
    """
    try:
        _, schedule_time = get_schedule_with_time(scheduler, wg, worker_counts, worker_names)
        return schedule_time <= deadline
    # TODO: remove
    except AssertionError:
        return False
    except Exception as ex:
        print(ex)
        return False


def find_min_workers(wg: WorkGraph, max_workers: int) -> tuple[np.array, list[str]]:
    """
    Find minimum amount of resource of certain type among each worker and build list of workers with minimum amount
    certain type of resource

    :param wg:
    :param max_workers:
    :return:
    """
    min_workers = defaultdict(lambda: max_workers)
    for node in wg.nodes:
        for req in node.work_unit.worker_reqs:
            min_workers[req.kind] = min(min_workers[req.kind], req.min_count)
    min_counts_list, worker_names = list(zip(*[(count, name) for name, count in min_workers.items()]))
    min_counts: np.array = np.array(min_counts_list, dtype=int)
    return min_counts, worker_names


def get_minimal_counts_by_schedule(scheduled_works: Schedule, worker_names: list[str]) -> np.array:
    workers_intervals = get_workers_intervals(scheduled_works)
    max_used_counts = defaultdict(int)
    for name_index in workers_intervals:
        name, index = name_index.split(SPLITTER)
        max_used_counts[name] = max(max_used_counts[name], int(index))
    optimal_counts = np.array([max_used_counts[name] + 1 for name in worker_names], dtype=int)
    return optimal_counts


def init_borders(wg: WorkGraph,
                 scheduler: Scheduler,
                 deadline: Time,
                 worker_factor: int,
                 max_workers: int,
                 right_workers: WorkerContractorPool or None) -> tuple[
    Optional[np.ndarray], Optional[np.ndarray], list[str]]:
    left_counts, worker_names = find_min_workers(wg, max_workers)
    if right_workers is None:
        right_counts: np.ndarray = left_counts * worker_factor
    else:
        right_counts = np.array([right_workers[name] for name in worker_names])

    if is_resources_good(wg, left_counts, worker_names, scheduler, deadline):
        # wg is resolved in time by minimal set of workers
        return left_counts, None, worker_names

    if not is_resources_good(wg, right_counts, worker_names, scheduler, deadline):
        # wg is not resolved in time by any set of workers
        return None, right_counts, worker_names

    return left_counts, right_counts, worker_names


def prepare_answer(counts: np.array,
                   worker_names: list[str],
                   wg: WorkGraph,
                   scheduler: Scheduler,
                   dry_resources: bool):
    schedule, max_time = get_schedule_with_time(scheduler, wg, counts, worker_names)
    if dry_resources:
        optimal_counts = get_minimal_counts_by_schedule(schedule, worker_names)
    else:
        optimal_counts = counts
    contractor = workers_to_contractors(optimal_counts, worker_names)[0]
    return contractor, max_time
