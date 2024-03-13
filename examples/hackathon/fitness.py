from typing import Callable

from sampo.api.genetic_api import ChromosomeType, FitnessFunction
from sampo.schemas.schedule import Schedule
from sampo.schemas.time import Time


def count_resources(schedule: Schedule) -> int:
    return len(set([worker.name for swork in schedule.works for worker in swork.workers]))


class MultiFitness(FitnessFunction):
    def __init__(self, consider_cost: bool = True, consider_resources: bool = False):
        self._cost = consider_cost
        self._resources = consider_resources

    def evaluate(self, chromosome: ChromosomeType, evaluator: Callable[[ChromosomeType], Schedule]) -> tuple[int, ...]:
        schedule = evaluator(chromosome)
        fitness = (Time.inf().value if schedule is None else schedule.execution_time.value,)
        if self._cost:
            fitness = (*fitness, Time.inf().value if schedule is None else schedule.pure_schedule_df['cost'].sum(),)
        if self._resources:
            fitness = (*fitness, Time.inf().value if schedule is None else count_resources(schedule))
        return fitness
