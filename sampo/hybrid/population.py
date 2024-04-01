from abc import ABC, abstractmethod

from sampo.api.genetic_api import ChromosomeType
from sampo.schemas import WorkGraph, Contractor, Time, LandscapeConfiguration
from sampo.schemas.schedule_spec import ScheduleSpec


class PopulationScheduler(ABC):

    @abstractmethod
    def schedule(self,
                 initial_population: list[ChromosomeType],
                 wg: WorkGraph,
                 contractors: list[Contractor],
                 spec: ScheduleSpec = ScheduleSpec(),
                 assigned_parent_time: Time = Time(0),
                 landscape: LandscapeConfiguration = LandscapeConfiguration()) -> list[ChromosomeType]:
        ...