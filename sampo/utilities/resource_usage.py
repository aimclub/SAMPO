from collections import defaultdict
from typing import Iterable

from sampo.schemas.schedule import Schedule
from sampo.schemas.time import Time


def get_total_resources_usage(schedule: Schedule) -> dict[str, list[int]]:
    usage = defaultdict(lambda: [0 for _ in range(schedule.execution_time.value + 1)])

    for work in schedule.works:
        for worker in work.workers:
            for day in range(work.start_time.value, work.finish_time.value + 1):
                usage[worker.name][day] += worker.count
    return usage


def get_resources_peak_usage(schedule: Schedule) -> dict[str, int]:
    return {res: max(res_usage) for res, res_usage in get_total_resources_usage(schedule).items()}


def resources_peaks_sum(schedule: Schedule) -> int:
    if schedule.execution_time.is_inf():
        return Time.inf().value
    return sum(get_resources_peak_usage(schedule).values())


def resources_sum(schedule: Schedule, resources_names: Iterable[str] | None = None) -> int:
    """
    Count the summary usage of resources in received schedule
    """
    is_none = resources_names is None
    resources_names = set(resources_names) if not is_none else {}

    res_sum = sum([sum([worker.count for worker in work.workers
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
