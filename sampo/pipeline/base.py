from abc import ABC, abstractmethod

import pandas as pd

from sampo.pipeline.lag_optimization import LagOptimizationStrategy
from sampo.scheduler.base import Scheduler
from sampo.scheduler.utils.local_optimization import OrderLocalOptimizer, ScheduleLocalOptimizer
from sampo.schemas.contractor import Contractor
from sampo.schemas.graph import WorkGraph, GraphNode
from sampo.schemas.landscape import LandscapeConfiguration
from sampo.schemas.project import ScheduledProject
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator
from sampo.utilities.name_mapper import NameMapper


class InputPipeline(ABC):
    """
    Base class to build different pipeline, that help to use the framework
    """

    @abstractmethod
    def wg(self, wg: WorkGraph | pd.DataFrame | str,
           all_connections: bool = False,
           change_connections_info: bool = False,
           sep: str = ',') -> 'InputPipeline':
        ...

    @abstractmethod
    def contractors(self, contractors: list[Contractor] | pd.DataFrame | str) -> 'InputPipeline':
        ...

    @abstractmethod
    def name_mapper(self, name_mapper: NameMapper | str) -> 'InputPipeline':
        ...

    @abstractmethod
    def history(self, history: pd.DataFrame | str,
                sep: str = ',') -> 'InputPipeline':
        ...

    @abstractmethod
    def landscape(self, landscape_config: LandscapeConfiguration) -> 'InputPipeline':
        ...

    @abstractmethod
    def spec(self, spec: ScheduleSpec) -> 'InputPipeline':
        ...

    @abstractmethod
    def time_shift(self, time: Time) -> 'InputPipeline':
        ...

    @abstractmethod
    def lag_optimize(self, lag_optimize: LagOptimizationStrategy) -> 'InputPipeline':
        """
        Mandatory argument. Shows should graph be lag-optimized or not.
        If not defined, pipeline should search the best variant of this argument in result.

        :param lag_optimize:
        :return: the pipeline object
        """
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
    def schedule(self, scheduler: Scheduler, validate: bool = False) -> 'SchedulePipeline':
        ...


class SchedulePipeline(ABC):
    """
    The part of pipeline, that manipulates with the whole entire schedule.
    """

    @abstractmethod
    def optimize_local(self, optimizer: ScheduleLocalOptimizer, area: range) -> 'SchedulePipeline':
        ...

    @abstractmethod
    def finish(self) -> list[ScheduledProject]:
        ...

    @abstractmethod
    def visualization(self, start_date: str) -> list['Visualization']:
        ...
