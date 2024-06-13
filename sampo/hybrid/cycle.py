import numpy as np

from sampo.api.genetic_api import FitnessFunction, ChromosomeType, ScheduleGenerationScheme
from sampo.base import SAMPO
from sampo.hybrid.population import PopulationScheduler
from sampo.scheduler.genetic import TimeFitness
from sampo.scheduler.genetic.schedule_builder import create_toolbox
from sampo.schemas import WorkGraph, Contractor, Time, LandscapeConfiguration, Schedule
from sampo.schemas.schedule_spec import ScheduleSpec


class CycleHybridScheduler:
    def __init__(self,
                 starting_scheduler: PopulationScheduler,
                 cycle_schedulers: list[PopulationScheduler],
                 fitness: FitnessFunction = TimeFitness(),
                 max_plateau_size: int = 2):
        self._starting_scheduler = starting_scheduler
        self._cycle_schedulers = cycle_schedulers
        self._fitness = fitness
        self._max_plateau_size = max_plateau_size

    def _get_population_fitness(self, pop: list[ChromosomeType]):
        # return best chromosome's fitness
        return min(SAMPO.backend.compute_chromosomes(self._fitness, pop))

    def _get_best_individual(self, pop: list[ChromosomeType]) -> ChromosomeType:
        fitness = SAMPO.backend.compute_chromosomes(self._fitness, pop)
        return pop[np.argmin(fitness)]

    def run(self,
            wg: WorkGraph,
            contractors: list[Contractor],
            spec: ScheduleSpec = ScheduleSpec(),
            assigned_parent_time: Time = Time(0),
            landscape: LandscapeConfiguration = LandscapeConfiguration()) -> ChromosomeType:
        pop = self._starting_scheduler.schedule([], wg, contractors, spec, assigned_parent_time, landscape)

        cur_fitness = Time.inf().value
        plateau_steps = 0

        while True:
            pop_fitness = self._get_population_fitness(pop)
            if pop_fitness == cur_fitness:
                plateau_steps += 1
                if plateau_steps == self._max_plateau_size:
                    break
            else:
                plateau_steps = 0
                cur_fitness = pop_fitness

            for scheduler in self._cycle_schedulers:
                pop = scheduler.schedule(pop, wg, contractors, spec, assigned_parent_time, landscape)

        return self._get_best_individual(pop)

    def schedule(self,
                 wg: WorkGraph,
                 contractors: list[Contractor],
                 spec: ScheduleSpec = ScheduleSpec(),
                 assigned_parent_time: Time = Time(0),
                 sgs_type: ScheduleGenerationScheme = ScheduleGenerationScheme.Parallel,
                 landscape: LandscapeConfiguration = LandscapeConfiguration()) -> Schedule:
        best_ind = self.run(wg, contractors, spec, assigned_parent_time, landscape)

        toolbox = create_toolbox(wg=wg, contractors=contractors, landscape=landscape,
                                 assigned_parent_time=assigned_parent_time, spec=spec,
                                 sgs_type=sgs_type)
        node2swork = toolbox.chromosome_to_schedule(best_ind)[0]

        return Schedule.from_scheduled_works(node2swork.values(), wg)
