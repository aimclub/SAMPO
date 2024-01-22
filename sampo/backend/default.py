from typing import Callable

from deap.base import Toolbox

from sampo.api.genetic_api import FitnessFunction, ChromosomeType
from sampo.backend import ComputationalBackend, T, R, ComputationalContext
from sampo.schemas import WorkGraph, Contractor, LandscapeConfiguration
from sampo.schemas.schedule_spec import ScheduleSpec


class DefaultComputationalContext(ComputationalContext):

    def map(self, action: Callable[[T], R], values: list[T]) -> list[R]:
        return [action(v) for v in values]


class DefaultComputationalBackend(ComputationalBackend):

    def new_context(self) -> ComputationalContext:
        return DefaultComputationalContext()

    def cache_scheduler_info(self, wg: WorkGraph, contractors: list[Contractor], landscape: LandscapeConfiguration,
                             spec: ScheduleSpec, toolbox: Toolbox):
        self._toolbox = toolbox

    def compute_chromosomes(self, fitness: FitnessFunction, chromosomes: list[ChromosomeType]) -> list[float]:
        return [fitness.evaluate(chromosome, self._toolbox.evaluate_chromosome) for chromosome in chromosomes]
