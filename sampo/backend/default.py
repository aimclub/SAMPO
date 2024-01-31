from random import Random
from typing import Callable

import sampo.scheduler

from sampo.api.genetic_api import FitnessFunction, ChromosomeType, Individual
from sampo.backend import ComputationalBackend, T, R
from sampo.schemas import WorkGraph, Contractor, LandscapeConfiguration, WorkTimeEstimator, Schedule, GraphNode, Time
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.schemas.time_estimator import DefaultWorkEstimator


class DefaultComputationalBackend(ComputationalBackend):

    def map(self, action: Callable[[T], R], values: list[T]) -> list[R]:
        return [action(v) for v in values]

    def cache_scheduler_info(self,
                             wg: WorkGraph,
                             contractors: list[Contractor],
                             landscape: LandscapeConfiguration,
                             spec: ScheduleSpec,
                             rand: Random | None = None,
                             work_estimator: WorkTimeEstimator | None = None):
        self._wg = wg
        self._contractors = contractors
        self._landscape = landscape
        self._spec = spec
        self._rand = rand
        self._work_estimator = work_estimator

    def cache_genetic_info(self,
                           population_size: int,
                           mutate_order: float,
                           mutate_resources: float,
                           mutate_zones: float,
                           deadline: Time | None,
                           weights: list[int] | None,
                           init_schedules: dict[str, tuple[Schedule, list[GraphNode] | None, ScheduleSpec, float]],
                           assigned_parent_time: Time):
        self._selection_size = population_size
        self._mutate_order = mutate_order
        self._mutate_resources = mutate_resources
        self._mutate_zones = mutate_zones
        self._deadline = deadline
        self._weights = weights
        self._init_schedules = init_schedules
        self._assigned_parent_time = assigned_parent_time

    def _ensure_toolbox_created(self):
        if self._toolbox is None:
            from sampo.scheduler.genetic.utils import init_chromosomes_f, create_toolbox_using_cached_chromosomes

            init_chromosomes = init_chromosomes_f(self._wg, self._contractors, self._init_schedules,
                                                  self._landscape)

            rand = self._rand or Random()
            work_estimator = self._work_estimator or DefaultWorkEstimator()
            assigned_parent_time = self._assigned_parent_time or Time(0)

            self._toolbox = create_toolbox_using_cached_chromosomes(self._wg,
                                                                    self._contractors,
                                                                    self._selection_size,
                                                                    self._mutate_order,
                                                                    self._mutate_resources,
                                                                    self._mutate_zones,
                                                                    init_chromosomes,
                                                                    rand,
                                                                    self._spec,
                                                                    work_estimator,
                                                                    assigned_parent_time,
                                                                    self._landscape)

    def compute_chromosomes(self,
                            fitness: FitnessFunction,
                            chromosomes: list[ChromosomeType]) -> list[float]:
        self._ensure_toolbox_created()
        return [fitness.evaluate(chromosome, self._toolbox.evaluate_chromosome) for chromosome in chromosomes]

    def generate_first_population(self, size_population: int) -> list[Individual]:
        self._ensure_toolbox_created()
        return self._toolbox.population(size_population)
