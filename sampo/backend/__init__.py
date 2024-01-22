from abc import ABC, abstractmethod
from typing import Callable, TypeVar

from deap.base import Toolbox

from sampo.api.genetic_api import ChromosomeType, FitnessFunction
from sampo.schemas import WorkGraph, Contractor, LandscapeConfiguration
from sampo.schemas.schedule_spec import ScheduleSpec

T = TypeVar('T')
R = TypeVar('R')

class ComputationalContext(ABC):
    @abstractmethod
    def map(self, action: Callable[[T], R], values: list[T]) -> list[R]:
        ...



class ComputationalBackend(ABC):

    def __init__(self):
        self._context = self.new_context()
        self._actions = {}

    def map(self, action: Callable[[T], R], values: list[T]) -> list[R]:
        return self._context.map(action, values)

    def set_context(self, context: ComputationalContext):
        self._context = context

    @abstractmethod
    def new_context(self) -> ComputationalContext:
        ...

    @abstractmethod
    def cache_scheduler_info(self,
                             wg: WorkGraph,
                             contractors: list[Contractor],
                             landscape: LandscapeConfiguration,
                             spec: ScheduleSpec,
                             toolbox: Toolbox):
        ...

    @abstractmethod
    def compute_chromosomes(self, fitness: FitnessFunction, chromosomes: list[ChromosomeType]) -> list[float]:
        ...

from sampo.backend.default import DefaultComputationalBackend
