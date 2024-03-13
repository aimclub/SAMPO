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


class WeightedFitness(FitnessFunction):
    def __init__(self, time_weight: float = 0.5, cost_weight: float = 0.5, resources_weight: float = 0):
        self._time = time_weight
        self._cost = cost_weight
        self._resources = resources_weight

    def evaluate(self, chromosome: ChromosomeType, evaluator: Callable[[ChromosomeType], Schedule]) -> tuple[int]:
        schedule = evaluator(chromosome)
        if schedule is None:
            return (Time.inf().value,)
        return (self._time * schedule.execution_time + self._cost * schedule.pure_schedule_df['cost'].sum()
                + self._resources * count_resources(schedule),)

