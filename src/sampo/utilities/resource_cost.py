from sampo.schemas.schedule import Schedule


def schedule_cost(schedule: Schedule) -> float:
    cost: float = 0

    for work in schedule.works:
        for worker in work.workers:
            cost += worker.get_cost() * work.duration.value
    return cost
