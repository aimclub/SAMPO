from random import Random

from sampo.scheduler.base import Scheduler, SchedulerType
from sampo.scheduler.genetic.utils import init_chromosomes_f, create_toolbox_using_cached_chromosomes
from sampo.scheduler.native_wrapper import NativeWrapper

from sampo.schemas import WorkGraph, Contractor, LandscapeConfiguration, WorkTimeEstimator, Schedule, Time, GraphNode
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.schemas.time_estimator import DefaultWorkEstimator

from sampo.api.genetic_api import FitnessFunction, ChromosomeType
from sampo.backend import ComputationalBackend, DefaultComputationalBackend, BackendActions
from sampo.backend.multiproc import MultiprocessingComputationalContext, MultiprocessingComputationalBackend


def cache_scheduler_info(backend: ComputationalBackend,
                         wg: WorkGraph,
                         contractors: list[Contractor],
                         landscape: LandscapeConfiguration,
                         spec: ScheduleSpec,
                         rand: Random | None = None,
                         work_estimator: WorkTimeEstimator | None = None):
    backend._wg = wg
    backend._contractors = contractors
    backend._landscape = landscape
    backend._spec = spec
    backend._rand = rand
    backend._work_estimator = work_estimator


def cache_genetic_info(backend: ComputationalBackend,
                       population_size: int,
                       mutate_order: float,
                       mutate_resources: float,
                       mutate_zones: float,
                       init_schedules: dict[str, tuple[Schedule, list[GraphNode] | None, ScheduleSpec, float]],
                       assigned_parent_time: Time):
    backend._population_size = population_size
    backend._mutate_order = mutate_order
    backend._mutate_resources = mutate_resources
    backend._mutate_zones = mutate_zones
    backend._init_schedules = init_schedules
    backend._assigned_parent_time = assigned_parent_time


def compute_chromosomes(backend: ComputationalBackend, fitness: FitnessFunction,
                        chromosomes: list[ChromosomeType]) -> list[float]:
    if backend._toolbox is None:
        init_chromosomes = init_chromosomes_f(backend._wg, backend._contractors, backend._init_schedules,
                                              backend._landscape)

        rand = backend._rand or Random()
        work_estimator = backend._work_estimator or DefaultWorkEstimator()
        assigned_parent_time = backend._assigned_parent_time or Time(0)

        backend._toolbox = create_toolbox_using_cached_chromosomes(backend._wg,
                                                                   backend._contractors,
                                                                   backend._population_size,
                                                                   backend._mutate_order,
                                                                   backend._mutate_resources,
                                                                   backend._mutate_zones,
                                                                   init_chromosomes,
                                                                   rand,
                                                                   backend._spec,
                                                                   work_estimator,
                                                                   assigned_parent_time,
                                                                   backend._landscape)

    return [fitness.evaluate(chromosome, backend._toolbox.evaluate_chromosome) for chromosome in chromosomes]


DefaultComputationalBackend.register(BackendActions.CACHE_SCHEDULER_INFO, cache_scheduler_info)
DefaultComputationalBackend.register(BackendActions.CACHE_GENETIC_INFO, cache_genetic_info)
DefaultComputationalBackend.register(BackendActions.COMPUTE_CHROMOSOMES, compute_chromosomes)


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


def recreate_pool(self):
    self.set_context(MultiprocessingComputationalContext(self._n_cpus,
                                                         scheduler_info_initializer,
                                                         (self._wg,
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
                                                          self._work_estimator.get_recreate_info())))


def mp_cache_scheduler_info(backend: ComputationalBackend,
                            wg: WorkGraph,
                            contractors: list[Contractor],
                            landscape: LandscapeConfiguration,
                            spec: ScheduleSpec,
                            rand: Random | None = None,
                            work_estimator: WorkTimeEstimator = DefaultWorkEstimator()):
    cache_scheduler_info(backend, wg, contractors, landscape, spec, rand, work_estimator)
    recreate_pool(backend)


def mp_cache_genetic_info(backend: ComputationalBackend,
                          population_size: int,
                          mutate_order: float,
                          mutate_resources: float,
                          mutate_zones: float,
                          init_schedules: dict[str, tuple[Schedule, list[GraphNode] | None, ScheduleSpec, float]],
                          assigned_parent_time: Time):
    cache_genetic_info(backend, population_size, mutate_order, mutate_resources,
                       mutate_zones, init_schedules, assigned_parent_time)
    backend._init_chromosomes = init_chromosomes_f(backend._wg, backend._contractors,
                                                   init_schedules, backend._landscape)
    recreate_pool(backend)


def mp_compute_chromosomes(self, fitness: FitnessFunction, chromosomes: list[ChromosomeType]) -> list[float]:
    def mapper(chromosome):
        return fitness.evaluate(chromosome, g_toolbox.evaluate_chromosome)

    return self._context.map(mapper, chromosomes)


MultiprocessingComputationalBackend.register(BackendActions.CACHE_SCHEDULER_INFO, mp_cache_scheduler_info)
MultiprocessingComputationalBackend.register(BackendActions.CACHE_GENETIC_INFO, mp_cache_genetic_info)
MultiprocessingComputationalBackend.register(BackendActions.COMPUTE_CHROMOSOMES, mp_compute_chromosomes)
