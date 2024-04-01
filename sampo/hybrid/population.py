from abc import ABC, abstractmethod

from sampo.api.genetic_api import ChromosomeType
from sampo.scheduler import GeneticScheduler, Scheduler
from sampo.scheduler.genetic.schedule_builder import create_toolbox
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


class GeneticPopulationScheduler(PopulationScheduler):
    def __init__(self, genetic: GeneticScheduler):
        self._genetic = genetic

    def schedule(self,
                 initial_population: list[ChromosomeType],
                 wg: WorkGraph,
                 contractors: list[Contractor],
                 spec: ScheduleSpec = ScheduleSpec(),
                 assigned_parent_time: Time = Time(0),
                 landscape: LandscapeConfiguration = LandscapeConfiguration()) -> list[ChromosomeType]:
        return self._genetic.upgrade_pop(wg, contractors, initial_population, spec,
                                         assigned_parent_time, landscape=landscape)


class HeuristicPopulationScheduler(PopulationScheduler):
    def __init__(self, schedulers: list[Scheduler]):
        self._schedulers = schedulers

    def schedule(self,
                 initial_population: list[ChromosomeType],
                 wg: WorkGraph,
                 contractors: list[Contractor],
                 spec: ScheduleSpec = ScheduleSpec(),
                 assigned_parent_time: Time = Time(0),
                 landscape: LandscapeConfiguration = LandscapeConfiguration()) -> list[ChromosomeType]:
        toolbox = create_toolbox(wg=wg, contractors=contractors,
                                 spec=spec, assigned_parent_time=assigned_parent_time,
                                 landscape=landscape)
        return [toolbox.schedule_to_chromosome(scheduler.schedule(wg, contractors, spec, landscape=landscape))
                for scheduler in self._schedulers]
