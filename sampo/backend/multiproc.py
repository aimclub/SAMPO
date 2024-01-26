from typing import Callable

import sampo.scheduler

from random import Random

import pathos.multiprocessing

from sampo.api.genetic_api import ChromosomeType, FitnessFunction
from sampo.backend import T, R
from sampo.backend.default import DefaultComputationalBackend
from sampo.scheduler.genetic.utils import create_toolbox_using_cached_chromosomes, init_chromosomes_f
from sampo.schemas import WorkGraph, Contractor, LandscapeConfiguration, Time, WorkTimeEstimator, Schedule, GraphNode
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.schemas.time_estimator import DefaultWorkEstimator





def scheduler_info_initializer(wg: WorkGraph,
                               contractors: list[Contractor],
                               landscape: LandscapeConfiguration,
                               spec: ScheduleSpec,
                               population_size: int,
                               mutate_order: float,
                               mutate_resources: float,
                               mutate_zones: float,
                               init_chromosomes: dict[str, tuple[ChromosomeType, float, ScheduleSpec]],
                               assigned_parent_time: Time,
                               rand: Random | None = None,
                               work_estimator_recreate_params: tuple | None = None):
    global g_wg, g_contractors, g_landscape, g_spec, g_toolbox

    g_wg = wg
    g_contractors = contractors
    g_landscape = landscape
    g_spec = spec

    rand = rand or Random()
    assigned_parent_time = assigned_parent_time or Time(0)

    work_estimator = work_estimator_recreate_params[0](*work_estimator_recreate_params[1])

    if init_chromosomes is not None:
        g_toolbox = create_toolbox_using_cached_chromosomes(wg,
                                                            contractors,
                                                            population_size,
                                                            mutate_order,
                                                            mutate_resources,
                                                            mutate_zones,
                                                            init_chromosomes,
                                                            rand,
                                                            spec,
                                                            work_estimator,
                                                            assigned_parent_time,
                                                            landscape)


class MultiprocessingComputationalBackend(DefaultComputationalBackend):
    def __init__(self, n_cpus: int):
        self._n_cpus = n_cpus
        self._init_chromosomes = None
        super().__init__()

    def map(self, action: Callable[[T], R], values: list[T]) -> list[R]:
        return self._pool.map(action, values)

    def _recreate_pool(self):
        self._pool = pathos.multiprocessing.Pool(self._n_cpus,
                                                 initializer=scheduler_info_initializer,
                                                 initargs=(self._wg,
                                                           self._contractors,
                                                           self._landscape,
                                                           self._spec,
                                                           self._population_size,
                                                           self._mutate_order,
                                                           self._mutate_resources,
                                                           self._mutate_zones,
                                                           self._init_chromosomes,
                                                           self._assigned_parent_time,
                                                           self._rand,
                                                           self._work_estimator.get_recreate_info()))

    def cache_scheduler_info(self,
                             wg: WorkGraph,
                             contractors: list[Contractor],
                             landscape: LandscapeConfiguration,
                             spec: ScheduleSpec,
                             rand: Random | None = None,
                             work_estimator: WorkTimeEstimator = DefaultWorkEstimator()):
        super().cache_scheduler_info(wg, contractors, landscape, spec, rand, work_estimator)
        self._recreate_pool()

    def cache_genetic_info(self,
                           population_size: int,
                           mutate_order: float,
                           mutate_resources: float,
                           mutate_zones: float,
                           init_schedules: dict[str, tuple[Schedule, list[GraphNode] | None, ScheduleSpec, float]],
                           assigned_parent_time: Time):
        super().cache_genetic_info(population_size, mutate_order, mutate_resources,
                                   mutate_zones, init_schedules, assigned_parent_time)
        self._init_chromosomes = init_chromosomes_f(self._wg, self._contractors, init_schedules, self._landscape)
        self._recreate_pool()

    def compute_chromosomes(self, fitness: FitnessFunction, chromosomes: list[ChromosomeType]) -> list[float]:
        def mapper(chromosome):
            return fitness.evaluate(chromosome, g_toolbox.evaluate_chromosome)

        return self.map(mapper, chromosomes)
