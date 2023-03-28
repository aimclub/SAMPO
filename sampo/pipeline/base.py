from abc import ABC, abstractmethod

from sampo.scheduler.base import Scheduler
from sampo.scheduler.utils.local_optimization import OrderLocalOptimizer, ScheduleLocalOptimizer
from sampo.schemas.contractor import Contractor
from sampo.schemas.graph import WorkGraph, GraphNode
from sampo.schemas.schedule import Schedule
from sampo.schemas.time_estimator import WorkTimeEstimator


class InputPipeline(ABC):

    @abstractmethod
    def wg(self, wg: WorkGraph) -> 'InputPipeline':
        ...

    @abstractmethod
    def contractors(self, contractors: list[Contractor]) -> 'InputPipeline':
        ...

    @abstractmethod
    def work_estimator(self, work_estimator: WorkTimeEstimator) -> 'InputPipeline':
        ...

    @abstractmethod
    def node_order(self, node_order: list[GraphNode]) -> 'InputPipeline':
        ...

    @abstractmethod
    def optimize_local(self, optimizer: OrderLocalOptimizer, area: range) -> 'InputPipeline':
        ...

    @abstractmethod
    def schedule(self, scheduler: Scheduler) -> 'SchedulePipeline':
        ...


class SchedulePipeline(ABC):

    @abstractmethod
    def optimize_local(self, optimizer: ScheduleLocalOptimizer, area: range) -> 'SchedulePipeline':
        ...

    @abstractmethod
    def finish(self) -> Schedule:
        ...
