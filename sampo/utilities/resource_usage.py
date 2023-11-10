import numpy as np
from collections import defaultdict
from sortedcontainers import SortedList
from typing import Iterable

from sampo.schemas.schedule import Schedule
from sampo.schemas.time import Time


def get_total_resources_usage(schedule: Schedule, resources_names: Iterable[str] | None = None) -> dict[str, np.ndarray]:
    df = schedule.full_schedule_df
    points = df[['start', 'finish']].to_numpy().copy()
    points[:, 1] += 1
    points = SortedList(set(points.flatten()))
    usage = defaultdict(lambda: np.zeros_like(points))

    is_none = resources_names is None
    resources_names = set(resources_names) if not is_none else {}

    for swork in schedule.works:
        start = points.bisect_left(swork.start_time)
        finish = points.bisect_left(swork.finish_time + 1)
        for worker in swork.workers:
            if worker.name in resources_names or is_none:
                usage[worker.name][start: finish] += worker.count

    return usage


def get_resources_peak_usage(schedule: Schedule, resources_names: Iterable[str] | None = None) -> dict[str, int]:
    return {res: max(res_usage) for res, res_usage in get_total_resources_usage(schedule, resources_names).items()}


def resources_peaks_sum(schedule: Schedule, resources_names: Iterable[str] | None = None) -> int:
    """
    Count the summary of resources peaks usage in received schedule
    """
    if schedule.execution_time.is_inf():
        return Time.inf().value
    return sum(get_resources_peak_usage(schedule, resources_names).values())


def resources_sum(schedule: Schedule, resources_names: Iterable[str] | None = None) -> int:
    """
    Count the summary usage of resources in received schedule
    """
    is_none = resources_names is None
    resources_names = set(resources_names) if not is_none else {}

    res_sum = sum([sum([worker.count * work.duration.value for worker in work.workers
                        if worker.name in resources_names or is_none], start=0)
                   for work in schedule.works])

    return res_sum


def resources_costs_sum(schedule: Schedule, resources_names: Iterable[str] | None = None) -> float:
    """
    Count the summary cost of resources in received schedule
    """
    is_none = resources_names is None
    resources_names = set(resources_names) if not is_none else {}

    cost = sum([sum([worker.get_cost() * work.duration.value for worker in work.workers
                     if worker.name in resources_names or is_none], start=0.0)
                for work in schedule.works])

    return cost
