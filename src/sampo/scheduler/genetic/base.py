import random
from typing import Dict, List, Tuple, Optional

from sampo.scheduler.base import Scheduler, SchedulerType
from sampo.scheduler.genetic.schedule_builder import build_schedule
from sampo.scheduler.heft.base import HEFTScheduler
from sampo.scheduler.heft_between.base import HEFTBetweenScheduler
from sampo.scheduler.resource.identity import IdentityResourceOptimizer
from sampo.scheduler.timeline.base import Timeline
from sampo.schemas.contractor import Contractor, get_worker_contractor_pool
from sampo.schemas.graph import WorkGraph
from sampo.schemas.schedule import Schedule
from sampo.schemas.schedule_spec import ScheduleSpec
from sampo.schemas.time import Time
from sampo.schemas.time_estimator import WorkTimeEstimator
from sampo.utilities.validation import validate_schedule


class GeneticScheduler(Scheduler):

    def __init__(self,
                 number_of_generation: Optional[int] = 50,
                 size_selection: Optional[int or None] = None,
                 mutate_order: Optional[float or None] = None,
                 mutate_resources: Optional[float or None] = None,
                 size_of_population: Optional[float or None] = None,
                 rand: Optional[random.Random] = None,
                 seed: Optional[float or None] = None,
                 scheduler_type: SchedulerType = SchedulerType.Genetic,
                 work_estimator: Optional[WorkTimeEstimator or None] = None):
        super().__init__(scheduler_type=scheduler_type,
                         resource_optimizer=IdentityResourceOptimizer(),
                         work_estimator=work_estimator)
        self.number_of_generation = number_of_generation
        self.size_selection = size_selection
        self.mutate_order = mutate_order
        self.mutate_resources = mutate_resources
        self.size_of_population = size_of_population
        self.rand = rand or random.Random(seed)
        self.work_estimator = work_estimator

    def get_params(self, works_count: int) -> Tuple[int, float, float, int]:
        size_selection = self.size_selection
        if size_selection is None:
            if works_count < 300:
                size_selection = 20
            else:
                size_selection = works_count // 15

        mutate_order = self.mutate_order
        if mutate_order is None:
            if works_count < 300:
                mutate_order = 0.006
            else:
                mutate_order = 2 / works_count

        mutate_resources = self.mutate_resources
        if mutate_resources is None:
            if works_count < 300:
                mutate_resources = 0.06
            else:
                mutate_resources = 18 / works_count

        size_of_population = self.size_of_population
        if size_of_population is None:
            if works_count < 300:
                size_of_population = 80
            elif 1500 > works_count >= 300:
                size_of_population = 50
            else:
                size_of_population = works_count // 50
        return size_selection, mutate_order, mutate_resources, size_of_population

    def schedule_with_cache(self, wg: WorkGraph,
                            contractors: List[Contractor],
                            spec: ScheduleSpec = ScheduleSpec(),
                            validate: bool = False,
                            timeline: Timeline | None = None) \
            -> tuple[Schedule, Time, Timeline]:
        def init_schedule(scheduler_class):
            return scheduler_class(self.work_estimator).schedule(wg, contractors)

        init_schedules: Dict[str, Schedule] = {
            "heft_end": init_schedule(HEFTScheduler),
            "heft_between": init_schedule(HEFTBetweenScheduler)
        }

        size_selection, mutate_order, mutate_resources, size_of_population = self.get_params(wg.vertex_count)
        agents = get_worker_contractor_pool(contractors)

        scheduled_works, schedule_start_time, timeline = build_schedule(wg,
                                                                        contractors,
                                                                        agents,
                                                                        size_of_population,
                                                                        self.number_of_generation,
                                                                        size_selection,
                                                                        mutate_order,
                                                                        mutate_resources,
                                                                        init_schedules,
                                                                        self.rand,
                                                                        spec,
                                                                        self.work_estimator,
                                                                        timeline=timeline)
        schedule = Schedule.from_scheduled_works(scheduled_works.values(), wg)

        if validate:
            validate_schedule(schedule, wg, contractors)

        return schedule, schedule_start_time, timeline
