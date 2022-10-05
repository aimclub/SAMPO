import random
from typing import Dict, List, Tuple, Optional, Any

from schemas.work_estimator import WorkTimeEstimator
from scheduler.base import Scheduler, SchedulerType
from schemas.schedule import Schedule
from scheduler.evolution.schedule_builder import build_schedule
from schemas.contractor import Contractor, get_agents_from_contractors
from schemas.graph import WorkGraph
from utils.validation import check_validity_of_scheduling


class EvolutionScheduler(Scheduler):
    scheduler_type: SchedulerType = SchedulerType.Evolutionary

    def __init__(self, work_estimator: Optional[WorkTimeEstimator or None] = None,
                 number_of_generation: Optional[int] = 50,
                 size_selection: Optional[int or None] = None,
                 mutate_order: Optional[float or None] = None,
                 mutate_resources: Optional[float or None] = None,
                 size_of_population: Optional[float or None] = None,
                 rand: Optional[random.Random] = None,
                 seed: Optional[float or None] = None):
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

    def schedule(self, wg: WorkGraph,
                 contractors: List[Contractor],
                 ksg_info: Dict[str, Dict[str, Any]],
                 start: str,
                 validate_schedule: Optional[bool] = False) \
            -> Tuple[Schedule, List[str]]:

        init_schedules: Dict[str, Tuple[Schedule, List[str]]] = {}

        size_selection, mutate_order, mutate_resources, size_of_population = self.get_params(wg.vertex_count)
        agents = get_agents_from_contractors(contractors)

        scheduled_works = build_schedule(wg,
                                         contractors,
                                         agents,
                                         size_of_population,
                                         self.number_of_generation,
                                         size_selection,
                                         mutate_order,
                                         mutate_resources,
                                         init_schedules,
                                         self.rand,
                                         self.work_estimator)
        schedule = Schedule.from_scheduled_works(scheduled_works.values(), ksg_info, start)

        if validate_schedule:
            check_validity_of_scheduling(schedule, agents, wg)

        ordered_work_id = [item[0] for item in
                           sorted(scheduled_works.items(), key=lambda item: item[1].start_time)]
        return schedule, ordered_work_id
