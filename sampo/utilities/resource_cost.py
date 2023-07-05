from sampo.schemas.schedule import Schedule


def schedule_cost(schedule: Schedule) -> float:
    """
    Count the summary cost of received schedule

    :param schedule:
    :return:
    """
    cost: float = 0

    for work in schedule.works:
        cost += work.cost
    return cost
