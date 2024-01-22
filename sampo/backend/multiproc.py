from typing import Callable

import pathos.multiprocessing
from deap.base import Toolbox

from sampo.api.genetic_api import FitnessFunction, ChromosomeType
from sampo.backend import ComputationalBackend, T, R, ComputationalContext
from sampo.schemas import WorkGraph, Contractor, LandscapeConfiguration
from sampo.schemas.schedule_spec import ScheduleSpec

# logger = pathos.logger()
# logger.setLevel(0)

# context used for caching common data
class MultiprocessingComputationalContext(ComputationalContext):
    def __init__(self, n_cpus: int, initializer: Callable = None, args: tuple = ()):
        self._pool = pathos.multiprocessing.Pool(n_cpus, initializer=initializer, initargs=args)

    def map(self, action: Callable[[T], R], values: list[T]) -> list[R]:
        # return list(map(action, values))
        try:
            return self._pool.map(action, values)
        except Exception as e:
            #logger.debug(e)
            pass


def scheduler_info_initializer(wg: WorkGraph,
                               contractors: list[Contractor],
                               landscape: LandscapeConfiguration,
                               spec: ScheduleSpec,
                               toolbox: Toolbox):
    global g_wg, g_contractors, g_landscape, g_spec, g_toolbox
    g_wg = wg
    g_contractors = contractors
    g_landscape = landscape
    g_spec = spec
    g_toolbox = toolbox

    # logger.info('I\'m here!')

class MultiprocessingComputationalBackend(ComputationalBackend):

    def __init__(self, n_cpus: int):
        self._n_cpus = n_cpus
        super().__init__()

    def new_context(self) -> ComputationalContext:
        return MultiprocessingComputationalContext(self._n_cpus)

    def cache_scheduler_info(self,
                             wg: WorkGraph,
                             contractors: list[Contractor],
                             landscape: LandscapeConfiguration,
                             spec: ScheduleSpec,
                             toolbox: Toolbox):
        self.set_context(MultiprocessingComputationalContext(self._n_cpus,
                                                             scheduler_info_initializer,
                                                             (wg, contractors, landscape, spec, toolbox)))

    def compute_chromosomes(self, fitness: FitnessFunction, chromosomes: list[ChromosomeType]) -> list[float]:
        def mapper(chromosome):
            return fitness.evaluate(chromosome, g_toolbox.evaluate_chromosome)

        return self._context.map(mapper, chromosomes)
        # return [1 for _ in range(len(chromosomes))]
