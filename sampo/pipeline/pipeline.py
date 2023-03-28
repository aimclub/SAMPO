from abc import ABC, abstractmethod

from sampo.scheduler.base import Scheduler
from sampo.scheduler.utils.local_optimization import OrderLocalOptimizer, ScheduleLocalOptimizer
from sampo.schemas.schedule import Schedule


class InputPipeline(ABC):

    @abstractmethod
    def optimize_local(self, optimizer: OrderLocalOptimizer) -> 'InputPipeline':
        ...

    @abstractmethod
    def schedule(self, scheduler: Scheduler) -> 'SchedulePipeline':
        ...


class SchedulePipeline(ABC):

    @abstractmethod
    def optimize_local(self, optimizer: ScheduleLocalOptimizer) -> 'SchedulePipeline':
        ...

    @abstractmethod
    def finish(self) -> Schedule:
        ...
