from collections import defaultdict

from sampo.schemas.schedule import Schedule
from sampo.schemas.time import Time


def get_total_resource_usage(schedule: Schedule) -> dict[str, list[int]]:
    usage = defaultdict(lambda: [0 for _ in range(schedule.execution_time.value + 1)])

    for work in schedule.works:
        for worker in work.workers:
            for day in range(work.start_time.value, work.finish_time.value + 1):
                usage[worker.name][day] += worker.count
    return usage


def get_peak_resource_usage(schedule: Schedule) -> dict[str, int]:
    return {res: max(res_usage) for res, res_usage in get_total_resource_usage(schedule).items()}


def get_absolute_peak_resource_usage(schedule: Schedule) -> int:
    if schedule.execution_time.is_inf():
        return Time.inf().value
    return sum(get_peak_resource_usage(schedule).values())
