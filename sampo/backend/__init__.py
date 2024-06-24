from abc import ABC, abstractmethod
from random import Random
from typing import TypeVar

# import sampo.scheduler

from sampo.api.genetic_api import ChromosomeType, FitnessFunction, Individual, ScheduleGenerationScheme
from sampo.schemas import WorkGraph, Contractor, LandscapeConfiguration, Schedule, GraphNode, Time, WorkTimeEstimator
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.schemas.time_estimator import DefaultWorkEstimator

T = TypeVar('T')
R = TypeVar('R')


class ComputationalBackend(ABC):
    def __init__(self):
        # scheduler parameters
        self._wg = None
        self._contractors = None
        self._landscape = None
        self._spec = None
        self._rand = Random()
        self._work_estimator = DefaultWorkEstimator()

        # additional genetic parameters
        self._toolbox = None
        self._selection_size = None
        self._mutate_order = None
        self._mutate_resources = None
        self._mutate_zones = None
        self._deadline = None
        self._weights = None
        self._init_schedules = None
        self._assigned_parent_time = None
        self._fitness_weights = None
        self._sgs_type = None
        self._only_lft_initialization = None
        self._is_multiobjective = None

        from sampo.backend.default import DefaultComputationalBackend
        if self.__class__ != DefaultComputationalBackend:
            self._default = DefaultComputationalBackend()

    def cache_scheduler_info(self,
                             wg: WorkGraph,
                             contractors: list[Contractor],
                             landscape: LandscapeConfiguration = LandscapeConfiguration(),
                             spec: ScheduleSpec = ScheduleSpec(),
                             rand: Random | None = None,
                             work_estimator: WorkTimeEstimator  = DefaultWorkEstimator()):
        from sampo.base import SAMPO
        SAMPO.logger.debug(f'Function cache_scheduler_info for {self.__class__.__name__} '
                           f'is not implemented yet, setting fallback')
        return self._default.cache_scheduler_info(wg, contractors, landscape, spec, rand, work_estimator)

    def cache_genetic_info(self,
                           population_size: int = 50,,
                           mutate_order: float = 0.1,
                           mutate_resources: float = 0.05,
                           mutate_zones: float = 0.05,
                           deadline: Time | None = None,
                           weights: list[int] | None = None,
                           init_schedules: dict[str, tuple[Schedule, list[GraphNode] | None, ScheduleSpec, float]] = None,
                           assigned_parent_time: Time = Time(0),
                           fitness_weights: tuple[int | float, ...] = None,
                           sgs_type: ScheduleGenerationScheme = ScheduleGenerationScheme.Parallel,
                           only_lft_initialization: bool = False,
                           is_multiobjective: bool = False):
        from sampo.base import SAMPO
        SAMPO.logger.debug(f'Function cache_genetic_info for {self.__class__.__name__} '
                           f'is not implemented yet, setting fallback')
        return self._default.cache_genetic_info(population_size, mutate_order, mutate_resources, mutate_zones,
                                                deadline, weights, init_schedules, assigned_parent_time,
                                                fitness_weights, sgs_type, only_lft_initialization, is_multiobjective)

    def compute_chromosomes(self,
                            fitness: FitnessFunction,
                            chromosomes: list[ChromosomeType]) -> list[float]:
        from sampo.base import SAMPO
        SAMPO.logger.debug(f'Function compute_chromosomes for {self.__class__.__name__} '
                           f'is not implemented yet, setting fallback')
        return self._default.compute_chromosomes(fitness, chromosomes)

    def generate_first_population(self, size_population: int) -> list[Individual]:
        from sampo.base import SAMPO
        SAMPO.logger.debug(f'Function generate_first)population for {self.__class__.__name__} '
                           f'is not implemented yet, setting fallback')
        return self._default.generate_first_population(size_population)

    def compute_chromosomes_to_schedules(self, chromosomes: list[ChromosomeType]) -> list[Schedule]:
        from sampo.base import SAMPO
        SAMPO.logger.debug(f'Function generate_first)population for {self.__class__.__name__} '
                           f'is not implemented yet, setting fallback')
        return self._default.compute_chromosomes_to_schedules(chromosomes)
