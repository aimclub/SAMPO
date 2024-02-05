from abc import ABC, abstractmethod
from random import Random
from typing import TypeVar

# import sampo.scheduler

from sampo.api.genetic_api import ChromosomeType, FitnessFunction, Individual, ScheduleGenerationScheme
from sampo.schemas import WorkGraph, Contractor, LandscapeConfiguration, Schedule, GraphNode, Time, WorkTimeEstimator
from sampo.schemas.schedule_spec import ScheduleSpec

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
        self._work_estimator = None

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

    @abstractmethod
    def cache_scheduler_info(self,
                             wg: WorkGraph,
                             contractors: list[Contractor],
                             landscape: LandscapeConfiguration,
                             spec: ScheduleSpec,
                             rand: Random | None = None,
                             work_estimator: WorkTimeEstimator | None = None):
        ...

    @abstractmethod
    def cache_genetic_info(self,
                           population_size: int,
                           mutate_order: float,
                           mutate_resources: float,
                           mutate_zones: float,
                           deadline: Time | None,
                           weights: list[int] | None,
                           init_schedules: dict[str, tuple[Schedule, list[GraphNode] | None, ScheduleSpec, float]],
                           assigned_parent_time: Time,
                           fitness_weights: tuple[int | float, ...],
                           sgs_type: ScheduleGenerationScheme,
                           only_lft_initialization: bool,
                           is_multiobjective: bool):
        ...

    @abstractmethod
    def compute_chromosomes(self,
                            fitness: FitnessFunction,
                            chromosomes: list[ChromosomeType]) -> list[float]:
        ...

    @abstractmethod
    def generate_first_population(self, size_population: int) -> list[Individual]:
        ...
