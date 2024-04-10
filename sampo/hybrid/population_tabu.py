import numpy as np

from sampo.api.genetic_api import ChromosomeType, FitnessFunction
from sampo.base import SAMPO
from sampo.hybrid.population import PopulationScheduler
from sampo.scheduler.genetic import TimeFitness
from sampo.scheduler.utils import get_worker_contractor_pool
from sampo.schemas import WorkGraph, Contractor, Time, LandscapeConfiguration
from sampo.schemas.schedule_spec import ScheduleSpec

from sampo.scheduler.genetic.converter import ChromosomeType
from tabusearch.experiments.scheduling.fixtures import setup_toolbox

from tabusearch.experiments.scheduling.scheduling_utils import get_optimiser, OptimiserLifetime
from tabusearch.utility.chromosome import ChromosomeRW


class TabuPopulationScheduler(PopulationScheduler):

    def __init__(self, fitness: FitnessFunction = TimeFitness()):
        self._fitness = fitness

    def _get_population_fitness(self, pop: list[ChromosomeType]):
        # return best chromosome's fitness
        return min(SAMPO.backend.compute_chromosomes(self._fitness, pop))

    def _get_best_individual(self, pop: list[ChromosomeType], fitness: list[float] | None = None) -> ChromosomeType:
        fitness = fitness or SAMPO.backend.compute_chromosomes(self._fitness, pop)
        return pop[np.argmin(fitness)]

    def schedule(self,
                 initial_population: list[ChromosomeType],
                 wg: WorkGraph,
                 contractors: list[Contractor],
                 spec: ScheduleSpec = ScheduleSpec(),
                 assigned_parent_time: Time = Time(0),
                 landscape: LandscapeConfiguration = LandscapeConfiguration()) -> list[ChromosomeType]:
        tabu_toolbox = setup_toolbox(wg, contractors, get_worker_contractor_pool(contractors))

        leader = self._get_best_individual(initial_population)
        fitness = SAMPO.backend.compute_chromosomes(TimeFitness(), [leader])
        print(f'TABU initial fitness: {fitness}')

        tabu_leader = ChromosomeRW.from_sampo_chromosome(leader)

        opt_ord, opt_res = get_optimiser(tabu_toolbox,
                                         use_vp=True,
                                         optimisers_lifetime=OptimiserLifetime.Short)
        tabu_leader = opt_ord.optimize(tabu_leader)
        tabu_leader = opt_res.optimize(tabu_leader.position)

        chromosome = tabu_leader.position.to_sampo_chromosome()
        fitness = SAMPO.backend.compute_chromosomes(TimeFitness(), [chromosome])
        print(f'TABU fitness: {fitness}')

        return initial_population + [chromosome]
