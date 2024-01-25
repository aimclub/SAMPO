from abc import ABC, abstractmethod
from enum import Enum, auto
from random import Random
from typing import Callable, TypeVar, Any

from sampo.api.genetic_api import ChromosomeType, FitnessFunction
from sampo.schemas import WorkGraph, Contractor, LandscapeConfiguration, Schedule, GraphNode, Time, WorkTimeEstimator
from sampo.schemas.schedule_spec import ScheduleSpec

T = TypeVar('T')
R = TypeVar('R')


class BackendActions(Enum):
    CACHE_SCHEDULER_INFO = auto(),
    CACHE_GENETIC_INFO = auto(),
    COMPUTE_CHROMOSOMES = auto()


class ComputationalContext(ABC):
    @abstractmethod
    def map(self, action: Callable[[T], R], values: list[T]) -> list[R]:
        ...


class ComputationalBackend(ABC):
    def __init__(self):
        self._context = self.new_context()

        self._wg = None
        self._contractors = None
        self._landscape = None
        self._spec = None
        self._rand = None
        self._work_estimator = None

        self._toolbox = None
        self._population_size = None
        self._mutate_order = None
        self._mutate_resources = None
        self._mutate_zones = None
        self._init_schedules = None
        self._assigned_parent_time = None

    @classmethod
    def register(cls, action_type: BackendActions, action: Callable):
        cls._actions[action_type] = action

    def run(self, action_type: BackendActions, *args) -> Any:
        return self.__class__._actions[action_type](self, *args)

    def map(self, action: Callable[[T], R], values: list[T]) -> list[R]:
        return self._context.map(action, values)

    def set_context(self, context: ComputationalContext):
        self._context = context

    @abstractmethod
    def new_context(self) -> ComputationalContext:
        ...


from sampo.backend.default import DefaultComputationalBackend
