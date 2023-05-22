import math
import random
from typing import List, Tuple, Optional, Callable

from deap.base import Toolbox

from sampo.scheduler.base import Scheduler, SchedulerType
from sampo.scheduler.genetic.operators import FitnessFunction, TimeFitness
from sampo.scheduler.genetic.schedule_builder import build_schedule
from sampo.scheduler.heft.base import HEFTScheduler, HEFTBetweenScheduler
from sampo.scheduler.heft.prioritization import prioritization
from sampo.scheduler.resource.average_req import AverageReqResourceOptimizer
from sampo.scheduler.resource.base import ResourceOptimizer
from sampo.scheduler.resource.identity import IdentityResourceOptimizer
from sampo.scheduler.resources_in_time.average_binary_search import AverageBinarySearchResourceOptimizingScheduler
from sampo.scheduler.timeline.base import Timeline
from sampo.schemas.contractor import Contractor, get_worker_contractor_pool
from sampo.schemas.graph import WorkGraph, GraphNode
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
                 n_cpu: int = 1,
                 fitness_constructor: Callable[[Toolbox], FitnessFunction] = TimeFitness,
                 scheduler_type: SchedulerType = SchedulerType.Genetic,
                 resource_optimizer: ResourceOptimizer = IdentityResourceOptimizer(),
                 work_estimator: Optional[WorkTimeEstimator or None] = None):
        super().__init__(scheduler_type=scheduler_type,
                         resource_optimizer=resource_optimizer,
                         work_estimator=work_estimator)
        self.number_of_generation = number_of_generation
        self.size_selection = size_selection
        self.mutate_order = mutate_order
        self.mutate_resources = mutate_resources
        self.size_of_population = size_of_population
        self.rand = rand or random.Random(seed)
        self.fitness_constructor = fitness_constructor
        self.work_estimator = work_estimator
        self._n_cpu = n_cpu

        self._time_border = None
        self._deadline = None

    def __str__(self) -> str:
        return f'GeneticScheduler[' \
               f'generations={self.number_of_generation},' \
               f'size_selection={self.size_selection},' \
               f'mutate_order={self.mutate_order},' \
               f'mutate_resources={self.mutate_resources}' \
               f']'

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
                mutate_order = 0.05
            else:
                mutate_order = 2 / math.sqrt(works_count)

        mutate_resources = self.mutate_resources
        if mutate_resources is None:
            if works_count < 300:
                mutate_resources = 0.1
            else:
                mutate_resources = 6 / math.sqrt(works_count)

        size_of_population = self.size_of_population
        if size_of_population is None:
            if works_count < 300:
                size_of_population = 20
            elif 1500 > works_count >= 300:
                size_of_population = 50
            else:
                size_of_population = works_count // 50
        return size_selection, mutate_order, mutate_resources, size_of_population

    def set_use_multiprocessing(self, n_cpu: int):
        self._n_cpu = n_cpu

    def set_time_border(self, time_border: int):
        self._time_border = time_border

    def set_deadline(self, deadline: Time):
        self._deadline = deadline

    def generate_first_population(self, wg: WorkGraph, contractors: list[Contractor]):
        def init_k_schedule(scheduler_class, k):
            return (scheduler_class(work_estimator=self.work_estimator,
                                    resource_optimizer=AverageReqResourceOptimizer(k)).schedule(wg, contractors),
                    list(reversed(prioritization(wg, self.work_estimator))))

        if self._deadline is None:
            def init_schedule(scheduler_class):
                return (scheduler_class(work_estimator=self.work_estimator).schedule(wg, contractors),
                        list(reversed(prioritization(wg, self.work_estimator))))

            return {
                "heft_end": init_schedule(HEFTScheduler),
                "heft_between": init_schedule(HEFTBetweenScheduler),
                "12.5%": init_k_schedule(HEFTScheduler, 8),
                "25%": init_k_schedule(HEFTScheduler, 4),
                "75%": init_k_schedule(HEFTScheduler, 4 / 3),
                "87.5%": init_k_schedule(HEFTScheduler, 8 / 7)
            }
        else:
            def init_schedule(scheduler_class):
                schedule = AverageBinarySearchResourceOptimizingScheduler(
                    scheduler_class(work_estimator=self.work_estimator)
                ).schedule_with_cache(wg, contractors, self._deadline)[0]
                return schedule, list(reversed(prioritization(wg, self.work_estimator)))

            return {
                "heft_end": init_schedule(HEFTScheduler),
                "heft_between": init_schedule(HEFTBetweenScheduler),
                "12.5%": init_k_schedule(HEFTScheduler, 8),
                "25%": init_k_schedule(HEFTScheduler, 4),
                "75%": init_k_schedule(HEFTScheduler, 4 / 3),
                "87.5%": init_k_schedule(HEFTScheduler, 8 / 7)
            }

    def schedule_with_cache(self, wg: WorkGraph,
                            contractors: List[Contractor],
                            spec: ScheduleSpec = ScheduleSpec(),
                            validate: bool = False,
                            assigned_parent_time: Time = Time(0),
                            timeline: Timeline | None = None) \
            -> tuple[Schedule, Time, Timeline, list[GraphNode]]:

        init_schedules = self.generate_first_population(wg, contractors)

        size_selection, mutate_order, mutate_resources, size_of_population = self.get_params(wg.vertex_count)
        worker_pool = get_worker_contractor_pool(contractors)

        scheduled_works, schedule_start_time, timeline, order_nodes = build_schedule(wg,
                                                                                     contractors,
                                                                                     worker_pool,
                                                                                     size_of_population,
                                                                                     self.number_of_generation,
                                                                                     size_selection,
                                                                                     mutate_order,
                                                                                     mutate_resources,
                                                                                     init_schedules,
                                                                                     self.rand,
                                                                                     spec,
                                                                                     self.fitness_constructor,
                                                                                     self.work_estimator,
                                                                                     n_cpu=self._n_cpu,
                                                                                     assigned_parent_time=assigned_parent_time,
                                                                                     timeline=timeline,
                                                                                     time_border=self._time_border)
        schedule = Schedule.from_scheduled_works(scheduled_works.values(), wg)

        if validate:
            validate_schedule(schedule, wg, contractors)

        return schedule, schedule_start_time, timeline, order_nodes
