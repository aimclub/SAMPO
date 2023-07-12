from sampo.schemas.contractor import Contractor
from sampo.schemas.schedule import Schedule


def get_total_resource_usage(schedule: Schedule, contractors: list[Contractor]) -> dict[str, list[int]]:
    resources = list(set(res.name for contractor in contractors for res in contractor.workers.values()))
    usage = {res: [0 for _ in range(schedule.execution_time.value + 1)] for res in resources}

    for work in schedule.works:
        for worker in work.workers:
            for day in range(work.start_time.value, work.finish_time.value + 1):
                usage[worker.name][day] += worker.count
    return usage


def get_peak_resource_usage(schedule: Schedule, contractors: list[Contractor]) -> dict[str, int]:
    return {res: max(res_usage) for res, res_usage in get_total_resource_usage(schedule, contractors).items()}


def get_absolute_peak_resource_usage(schedule: Schedule, contractors: list[Contractor]) -> int:
    return sum(get_peak_resource_usage(schedule, contractors).values())
