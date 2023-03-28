from abc import ABC, abstractmethod

from sampo.scheduler.base import Scheduler
from sampo.scheduler.utils.local_optimization import OrderLocalOptimizer, ScheduleLocalOptimizer
from sampo.schemas.contractor import Contractor
from sampo.schemas.graph import WorkGraph
from sampo.schemas.schedule import Schedule


class InputPipeline(ABC):

    @abstractmethod
    def wg(self, wg: WorkGraph) -> 'InputPipeline':
        ...

    @abstractmethod
    def contractors(self, contractors: list[Contractor]) -> 'InputPipeline':
        ...

    @abstractmethod
    def optimize_local(self, optimizer: OrderLocalOptimizer, area: slice) -> 'InputPipeline':
        ...

    @abstractmethod
    def schedule(self, scheduler: Scheduler) -> 'SchedulePipeline':
        ...


class SchedulePipeline(ABC):

    @abstractmethod
    def optimize_local(self, optimizer: ScheduleLocalOptimizer, area: slice) -> 'SchedulePipeline':
        ...

    @abstractmethod
    def finish(self) -> Schedule:
        ...
